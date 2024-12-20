// Defines the struct Halo2Circuit, which implements the trait PlonkishCircuit.

use crate::{
    backend::{PlonkishCircuit, PlonkishCircuitInfo, WitnessEncoding},
    util::{
        arithmetic::{BatchInvert, Field},
        chain,
        expression::{Expression, Query, Rotation},
        izip, Itertools,
    },
};
use halo2_proofs::{
    circuit::Value,
    plonk::{
        self, Advice, Any, Assigned, Assignment, Challenge, Circuit, Column, ConstraintSystem,
        Error, Fixed, FloorPlanner, Instance, Selector,
    },
};
use rand::RngCore;
use std::{
    collections::{HashMap, HashSet},
    mem,
};

#[cfg(any(test, feature = "benchmark"))]
pub mod circuit;
#[cfg(test)]
mod test;


pub trait CircuitExt<F: Field>: Circuit<F> {
    fn rand(_k: usize, _rng: impl RngCore) -> Self
    where
        Self: Sized,
    {
        unimplemented!()
    }

    fn instances(&self) -> Vec<Vec<F>>;

    fn num_instances(&self) -> Vec<usize> {
        self.instances().iter().map(Vec::len).collect()
    }
}

pub struct Halo2Circuit<F: Field, C: Circuit<F>> {
    k: u32,
    instances: Vec<Vec<F>>,
    circuit: C,
    cs: ConstraintSystem<F>,
    config: C::Config,
    constants: Vec<Column<Fixed>>,
    num_witness_polys: Vec<usize>,
    advice_idx_in_phase: Vec<usize>,
    challenge_idx: Vec<usize>,
    row_mapping: Vec<usize>,
}


impl<F: Field, C: CircuitExt<F>> Halo2Circuit<F, C> {
    // Generate a new Halo2Circuit from a circuit of type C that implements CircuitExt trait, which extends the Circuit trait from halo2
    pub fn new<E: WitnessEncoding>(k: usize, circuit: C) -> Self {
        // Obtain constraint system cs and config from circuit 
        let (cs, config) = {
            let mut cs = ConstraintSystem::default();
            let config = C::configure(&mut cs);
            (cs, config)
        };
        // Obtain constants from constraint system
        let constants = cs.constants().clone();
        // Obtain the the number of witness polynomials for each phase (phase refers to multi phase challenges)
        let num_witness_polys = num_by_phase(&cs.advice_column_phase());
        // Obtain the index of each advice column within the phase
        let advice_idx_in_phase = idx_in_phase(&cs.advice_column_phase());
        // Order the challenges by phase
        let challenge_idx = idx_order_by_phase(&cs.challenge_phase(), 0);
        let row_mapping = E::row_mapping(k);

        Self {
            k: k as u32,
            instances: circuit.instances(),
            circuit,
            cs,
            config,
            constants,
            num_witness_polys,
            advice_idx_in_phase,
            challenge_idx,
            row_mapping,
        }
    }

    pub fn circuit(&self) -> &C {
        &self.circuit
    }

    pub fn update_witness(&mut self, f: impl FnOnce(&mut C)) {
        f(&mut self.circuit);
        self.instances = self.circuit.instances();
    }
}

impl<F: Field, C: Circuit<F>> AsRef<C> for Halo2Circuit<F, C> {
    fn as_ref(&self) -> &C {
        &self.circuit
    }
}

// Implement PlonkishCircuit trait from backend for Halo2Circuit struct 
impl<F: Field, C: Circuit<F>> PlonkishCircuit<F> for Halo2Circuit<F, C> {
    // Get PlonkishCircuitInfo from a Halo2 Circuit without preprocessing, i.e. setting fixed/ selector columns or copy constraints. 
    fn circuit_info_without_preprocess(&self) -> Result<PlonkishCircuitInfo<F>, crate::Error> {
        let Self {
            k,
            instances,
            cs,
            challenge_idx,
            ..
        } = self;
        // Generate indices of advice columns so that they are ordered by phase
        let advice_idx = advice_idx(cs);
        // Convert expressions from halo2 to backend 
        let constraints = cs
            .gates()
            .iter()
            .flat_map(|gate| {
                gate.polynomials().iter().map(|expression| {
                    convert_expression(cs, &advice_idx, challenge_idx, expression)
                })
            })
            .collect();
        // Convert lookup expressions from halo2 to backend 
        let lookups = cs
            .lookups()
            .iter()
            .map(|lookup| {
                lookup
                    .input_expressions()
                    .iter()
                    .zip(lookup.table_expressions())
                    .map(|(input, table)| {
                        let [input, table] = [input, table].map(|expression| {
                            convert_expression(cs, &advice_idx, challenge_idx, expression)
                        });
                        (input, table)
                    })
                    .collect_vec()
            })
            .collect();

        let num_instances = instances.iter().map(Vec::len).collect_vec();
        // Set preprocess_polys, initialized to 0, with 2^k rows. 
        let preprocess_polys =
            vec![vec![F::ZERO; 1 << k]; cs.num_selectors() + cs.num_fixed_columns()];
        //Obtain indices of columns from the contraint system 
        let column_idx = column_idx(cs);
        // Initialize permutations defining copy constraints 
        let permutations = cs
            .permutation()
            .get_columns()
            .iter()
            .map(|column| {
                let key = (*column.column_type(), column.index());
                vec![(column_idx[&key], 1)]
            })
            .collect_vec();

        Ok(PlonkishCircuitInfo {
            k: *k as usize,
            num_instances,
            preprocess_polys,
            num_witness_polys: num_by_phase(&cs.advice_column_phase()),
            num_challenges: num_by_phase(&cs.challenge_phase()),
            constraints,
            lookups,
            permutations,
            max_degree: Some(cs.degree::<false>()),
        })
    }

    // Get PlonkishCircuitInfo from a Halo2 Circuit with preprocessing, i.e. setting fixed/ selector columns or copy constraints. 
    fn circuit_info(&self) -> Result<PlonkishCircuitInfo<F>, crate::Error> {
        let Self {
            k,
            instances,
            cs,
            config,
            circuit,
            constants,
            row_mapping,
            ..
        } = self;
        let mut circuit_info = self.circuit_info_without_preprocess()?;

        let num_instances = instances.iter().map(Vec::len).collect_vec();
        //Obtain indices of columns from the contraint system 
        let column_idx = column_idx(cs);
        // Obtain initial permutations from constraint system, for now no copy constraints are set 
        let permutation_column_idx = cs
            .permutation()
            .get_columns()
            .iter()
            .map(|column| {
                let key = (*column.column_type(), column.index());
                (key, column_idx[&key])
            })
            .collect();
        // Initialize preprocess collector
        let mut preprocess_collector = PreprocessCollector {
            k: *k,
            num_instances,
            fixeds: vec![vec![F::ZERO.into(); 1 << k]; cs.num_fixed_columns()],
            permutation: Permutation::new(permutation_column_idx),
            selectors: vec![vec![false; 1 << k]; cs.num_selectors()],
            row_mapping,
        };

        // Use circuit defined in Halo2Circuit to set the circuit in preprocess collector
        C::FloorPlanner::synthesize(
            &mut preprocess_collector,
            circuit,
            config.clone(),
            constants.clone(),
        )
        .map_err(|err| crate::Error::InvalidSnark(format!("Synthesize failure: {err:?}")))?;

        // Obtain the fixed and selector columns from the preprocess collector
        circuit_info.preprocess_polys = chain![
            batch_invert_assigned(preprocess_collector.fixeds),
            preprocess_collector.selectors.into_iter().map(|selectors| {
                selectors
                    .into_iter()
                    .map(|selector| if selector { F::ONE } else { F::ZERO })
                    .collect()
            }),
        ]
        .collect();
    // Obtain the copy constraints from the preprocess collector
        circuit_info.permutations = preprocess_collector.permutation.into_cycles();

        Ok(circuit_info)
    }

    // Return instances
    fn instances(&self) -> &[Vec<F>] {
        &self.instances
    }


    // Synthesize a circuit for a particular phase with an instance and witness of the circuit,  and challenges
    fn synthesize(&self, phase: usize, challenges: &[F]) -> Result<Vec<Vec<F>>, crate::Error> {
        let instances = self.instances.iter().map(Vec::as_slice).collect_vec();
        let mut witness_collector = WitnessCollector {
            k: self.k,
            phase: phase as u8,
            advice_idx_in_phase: &self.advice_idx_in_phase,
            challenge_idx: &self.challenge_idx,
            instances: instances.as_slice(),
            // all advice columns are initialized to zero, there are 2^k rows and self.num_witness_polys[phase] columns 
            advices: vec![vec![F::ZERO.into(); 1 << self.k]; self.num_witness_polys[phase]],
            challenges,
            row_mapping: &self.row_mapping,
        };

        C::FloorPlanner::synthesize(
            &mut witness_collector,
            &self.circuit,
            self.config.clone(),
            self.constants.clone(),
        )
        .map_err(|err| crate::Error::InvalidSnark(format!("Synthesize failure: {err:?}")))?;

        Ok(batch_invert_assigned(witness_collector.advices))
    }
}

#[derive(Debug)]
struct PreprocessCollector<'a, F: Field> {
    k: u32,
    num_instances: Vec<usize>,
    fixeds: Vec<Vec<Assigned<F>>>,
    permutation: Permutation,
    selectors: Vec<Vec<bool>>,
    row_mapping: &'a [usize],
}

//Perform pre-processing for the circuit, i.e. setting fixed/ selector columns and copy constraints.
impl<'a, F: Field> Assignment<F> for PreprocessCollector<'a, F> {
    fn enter_region<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
    }

    fn exit_region(&mut self) {}

    fn annotate_column<A, AR>(&mut self, _: A, _: Column<Any>)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
    }

    // Set the values of the selector columns
    fn enable_selector<A, AR>(&mut self, _: A, selector: &Selector, row: usize) -> Result<(), Error>
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let Some(row) = self.row_mapping.get(row).copied() else {
            return Err(Error::NotEnoughRowsAvailable { current_k: self.k });
        };

        self.selectors[selector.index()][row] = true;

        Ok(())
    }

    // The instance is not set in preprocessing so unless the cell is out of bounds return Value::unknown.
    fn query_instance(&self, column: Column<Instance>, row: usize) -> Result<Value<F>, Error> {
        self.num_instances
            .get(column.index())
            .and_then(|num_instances| (row < *num_instances).then(Value::unknown))
            .ok_or(Error::BoundsFailure)
    }

    // Advice columns are set later in WitnessCollector
    fn assign_advice<V, VR, A, AR>(
        &mut self,
        _: A,
        _: Column<Advice>,
        _: usize,
        _: V,
    ) -> Result<(), Error>
    where
        V: FnOnce() -> Value<VR>,
        VR: Into<Assigned<F>>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        Ok(())
    }

    // Set the values of the fixed columns
    fn assign_fixed<V, VR, A, AR>(
        &mut self,
        _: A,
        column: Column<Fixed>,
        row: usize,
        to: V,
    ) -> Result<(), Error>
    where
        V: FnOnce() -> Value<VR>,
        VR: Into<Assigned<F>>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let Some(row) = self.row_mapping.get(row).copied() else {
            return Err(Error::NotEnoughRowsAvailable { current_k: self.k });
        };

        *self
            .fixeds
            .get_mut(column.index())
            .and_then(|v| v.get_mut(row))
            .ok_or(Error::BoundsFailure)? = to().into_field().assign()?;

        Ok(())
    }

    // Add a copy constraint to the permutation
    fn copy(
        &mut self,
        lhs_column: Column<Any>,
        lhs_row: usize,
        rhs_column: Column<Any>,
        rhs_row: usize,
    ) -> Result<(), Error> {
        let Some(lhs_row) = self.row_mapping.get(lhs_row).copied() else {
            return Err(Error::NotEnoughRowsAvailable { current_k: self.k });
        };
        let Some(rhs_row) = self.row_mapping.get(rhs_row).copied() else {
            return Err(Error::NotEnoughRowsAvailable { current_k: self.k });
        };
        self.permutation
            .copy(lhs_column, lhs_row, rhs_column, rhs_row)
    }

    // Set cells in a fixed column for all rows from from_row to the end
    fn fill_from_row(
        &mut self,
        column: Column<Fixed>,
        from_row: usize,
        to: Value<Assigned<F>>,
    ) -> Result<(), Error> {
        let Some(_) = self.row_mapping.get(from_row) else {
            return Err(Error::NotEnoughRowsAvailable { current_k: self.k });
        };

        let col = self
            .fixeds
            .get_mut(column.index())
            .ok_or(Error::BoundsFailure)?;

        let filler = to.assign()?;
        for row in self.row_mapping.iter().skip(from_row).copied() {
            col[row] = filler;
        }

        Ok(())
    }

    // Challenges are created later 
    fn get_challenge(&self, _: Challenge) -> Value<F> {
        Value::unknown()
    }

    fn push_namespace<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
    }

    fn pop_namespace(&mut self, _: Option<String>) {}
}

//Stores copy constraints in a permutation
#[derive(Debug)]
struct Permutation {
    column_idx: HashMap<(Any, usize), usize>,
    cycles: Vec<HashSet<(usize, usize)>>,
    cycle_idx: HashMap<(usize, usize), usize>,
}

impl Permutation {
    fn new(column_idx: HashMap<(Any, usize), usize>) -> Self {
        Self {
            column_idx,
            cycles: Default::default(),
            cycle_idx: Default::default(),
        }
    }
    //Add copy constraint
    fn copy(
        &mut self,
        lhs_column: Column<Any>,
        lhs_row: usize,
        rhs_column: Column<Any>,
        rhs_row: usize,
    ) -> Result<(), Error> {
        // Get column indices in the matrix
        let lhs_idx = *self
            .column_idx
            .get(&(*lhs_column.column_type(), lhs_column.index()))
            .ok_or(Error::ColumnNotInPermutation(lhs_column))?;
        let rhs_idx = *self
            .column_idx
            .get(&(*rhs_column.column_type(), rhs_column.index()))
            .ok_or(Error::ColumnNotInPermutation(rhs_column))?;

        match (
            self.cycle_idx.get(&(lhs_idx, lhs_row)).copied(),
            self.cycle_idx.get(&(rhs_idx, rhs_row)).copied(),
        ) {
            (Some(lhs_cycle_idx), Some(rhs_cycle_idx)) => {
                // Merge the two cycles together
                for cell in self.cycles[rhs_cycle_idx].iter().copied() {
                    self.cycle_idx.insert(cell, lhs_cycle_idx);
                }
                let rhs_cycle = mem::take(&mut self.cycles[rhs_cycle_idx]);
                self.cycles[lhs_cycle_idx].extend(rhs_cycle);
            }
            // If only one of the cells are in a cycle, add both cells to that cycle. If neither of the cells are in a cycle, add them to a new cycle
            cycle_idx => {
                let cycle_idx = if let (Some(cycle_idx), None) | (None, Some(cycle_idx)) = cycle_idx
                {
                    cycle_idx
                } else {
                    let cycle_idx = self.cycles.len();
                    self.cycles.push(Default::default());
                    cycle_idx
                };
                for cell in [(lhs_idx, lhs_row), (rhs_idx, rhs_row)] {
                    self.cycles[cycle_idx].insert(cell);
                    self.cycle_idx.insert(cell, cycle_idx);
                }
            }
        };

        Ok(())
    }

    fn into_cycles(self) -> Vec<Vec<(usize, usize)>> {
        self.cycles
            .into_iter()
            .filter_map(|cycle| {
                (!cycle.is_empty()).then(|| cycle.into_iter().sorted().collect_vec())
            })
            .collect()
    }
}

// This assigns the witness values to the circuit
#[derive(Debug)]
struct WitnessCollector<'a, F: Field> {
    k: u32,
    phase: u8,
    advice_idx_in_phase: &'a [usize],
    challenge_idx: &'a [usize],
    instances: &'a [&'a [F]],
    advices: Vec<Vec<Assigned<F>>>,
    challenges: &'a [F],
    row_mapping: &'a [usize],
}

impl<'a, F: Field> Assignment<F> for WitnessCollector<'a, F> {
    fn enter_region<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
    }

    fn exit_region(&mut self) {}

    fn annotate_column<A, AR>(&mut self, _: A, _: Column<Any>)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
    }

    // Selectors are enabled earlier in PreprocessCollector
    fn enable_selector<A, AR>(&mut self, _: A, _: &Selector, _: usize) -> Result<(), Error>
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        Ok(())
    }

    // Return cell from instance column
    fn query_instance(&self, column: Column<Instance>, row: usize) -> Result<Value<F>, Error> {
        self.instances
            .get(column.index())
            .and_then(|column| column.get(row))
            .map(|v| Value::known(*v))
            .ok_or(Error::BoundsFailure)
    }

    //Assign witness to circuit
    fn assign_advice<V, VR, A, AR>(
        &mut self,
        _: A,
        column: Column<Advice>,
        row: usize,
        to: V,
    ) -> Result<(), Error>
    where
        V: FnOnce() -> Value<VR>,
        VR: Into<Assigned<F>>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        //wait until correct phase to assign the advice column
        if self.phase != column.column_type().phase() {
            return Ok(());
        }

        let Some(row) = self.row_mapping.get(row).copied() else {
            return Err(Error::NotEnoughRowsAvailable { current_k: self.k });
        };

        *self
            .advices
            .get_mut(self.advice_idx_in_phase[column.index()])
            .and_then(|v| v.get_mut(row))
            .ok_or(Error::BoundsFailure)? = to().into_field().assign()?;

        Ok(())
    }

    // Fixed columns are set earlier in PreprocessCollector
    fn assign_fixed<V, VR, A, AR>(
        &mut self,
        _: A,
        _: Column<Fixed>,
        _: usize,
        _: V,
    ) -> Result<(), Error>
    where
        V: FnOnce() -> Value<VR>,
        VR: Into<Assigned<F>>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        Ok(())
    }

    // Copy Constraints are enabled earlier in PreprocessCollector
    fn copy(&mut self, _: Column<Any>, _: usize, _: Column<Any>, _: usize) -> Result<(), Error> {
        Ok(())
    }

    // Fixed columns are set earlier in PreprocessCollector
    fn fill_from_row(
        &mut self,
        _: Column<Fixed>,
        _: usize,
        _: Value<Assigned<F>>,
    ) -> Result<(), Error> {
        Ok(())
    }

    fn get_challenge(&self, challenge: Challenge) -> Value<F> {
        self.challenges
            .get(self.challenge_idx[challenge.index()])
            .copied()
            .map(Value::known)
            .unwrap_or_else(Value::unknown)
    }

    fn push_namespace<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
    }

    fn pop_namespace(&mut self, _: Option<String>) {}
}


// Output index of each advice column in the matrix such that columns are ordered by phase (with all advice columms at the end of the matrix)
fn advice_idx<F: Field>(cs: &ConstraintSystem<F>) -> Vec<usize> {
    // Order cs.advice_column_phase() such that cs.advice_column_phase()[i] <= cs.advice_column_phase()[i+1] and add the new index to the advice_offset (total number of columns - num advice columns) for each element.
    let advice_offset = cs.num_instance_columns() + cs.num_fixed_columns() + cs.num_selectors();
    idx_order_by_phase(&cs.advice_column_phase(), advice_offset)
}

//On input the constraint system output (HashMap(label,index1), index2) where label denotes whether its a Instance/ fixed or advice column, index1 is the original index in the constraint system,  and index2 is the new index of the column in the matrix (with advice columns ordered by phase))
fn column_idx<F: Field>(cs: &ConstraintSystem<F>) -> HashMap<(Any, usize), usize> {
    let advice_idx = advice_idx(cs);
    chain![
        (0..cs.num_instance_columns()).map(|idx| (Any::Instance, idx)),
        (0..cs.num_fixed_columns() + cs.num_selectors()).map(|idx| (Any::Fixed, idx)),
    ]
    .enumerate()
    .map(|(idx, column)| (column, idx))
    .chain((0..advice_idx.len()).map(|idx| ((Any::advice(), idx), advice_idx[idx])))
    .collect()
}

// Return the total number of phases, i.e. the max value in phases vector +1
fn num_phases(phases: &[u8]) -> usize {
    phases.iter().max().copied().unwrap_or_default() as usize + 1
}

// Outputs the number of elements in the phases vector for each phase
fn num_by_phase(phases: &[u8]) -> Vec<usize> {
    phases.iter().copied().fold(
        vec![0usize; num_phases(phases)],
        |mut num_by_phase, phase| {
            num_by_phase[phase as usize] += 1;
            num_by_phase
        },
    )
}

// For each advice column in phase j, output k such that this is the kth advice column in phase j
fn idx_in_phase(phases: &[u8]) -> Vec<usize> {
    phases
        .iter()
        .copied()
        .scan(vec![0; num_phases(phases)], |state, phase| {
            let index = state[phase as usize];
            state[phase as usize] += 1;
            Some(index)
        })
        .collect_vec()
}

//Output new order of phases such that phases[i] <= phases[i+1] and add the new index to the offset for each element
fn idx_order_by_phase(phases: &[u8], offset: usize) -> Vec<usize> {
    phases
        .iter()
        .copied()
        .scan(phase_offsets(phases), |state, phase| {
            let index = state[phase as usize];
            state[phase as usize] += 1;
            Some(offset + index)
        })
        .collect()
}

// returns output[0] =0,  output[i] = |j :  phases[j] <= i -1 |
fn phase_offsets(phases: &[u8]) -> Vec<usize> {
    num_by_phase(phases)
        .into_iter()
        .scan(0, |state, num| {
            let offset = *state;
            *state += num;
            Some(offset)
        })
        .collect()
}

//convert from an expression defined in halo2 to an expression defined in backend 
fn convert_expression<F: Field>(
    cs: &ConstraintSystem<F>,
    advice_idx: &[usize],
    challenge_idx: &[usize],
    expression: &plonk::Expression<F>,
) -> Expression<F> {
    expression.evaluate(
        &|constant| Expression::Constant(constant),
        &|selector| {
            let poly = cs.num_instance_columns() + cs.num_fixed_columns() + selector.index();
            Query::new(poly, Rotation::cur()).into()
        },
        &|query| {
            let poly = cs.num_instance_columns() + query.column_index();
            Query::new(poly, Rotation(query.rotation().0)).into()
        },
        &|query| {
            let poly = advice_idx[query.column_index()];
            Query::new(poly, Rotation(query.rotation().0)).into()
        },
        &|query| Query::new(query.column_index(), Rotation(query.rotation().0)).into(),
        &|challenge| Expression::Challenge(challenge_idx[challenge.index()]),
        &|value| -value,
        &|lhs, rhs| lhs + rhs,
        &|lhs, rhs| lhs * rhs,
        &|value, scalar| value * scalar,
    )
}

//For a matrix of elememts a/b return a matric of elements a b^-1
fn batch_invert_assigned<F: Field>(assigneds: Vec<Vec<Assigned<F>>>) -> Vec<Vec<F>> {
    // collect all the denominators in Assigned together into denoms
    let mut denoms: Vec<_> = assigneds
        .iter()
        .map(|f| {
            f.iter()
                .map(|value| value.denominator())
                .collect::<Vec<_>>()
        })
        .collect();

    denoms
        .iter_mut()
        .flat_map(|f| f.iter_mut().filter_map(|d| d.as_mut()))
        .batch_invert();

    izip!(&assigneds, denoms)
        .map(|(assigneds, denoms)| {
            assigneds
                .iter()
                .zip(denoms)
                .map(|(assigned, denom)| {
                    denom
                        .map(|denom| assigned.numerator() * denom)
                        .unwrap_or_else(|| assigned.numerator())
                })
                .collect()
        })
        .collect()
}
