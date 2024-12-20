use crate::{
    backend::{
        hyperplonk::{HyperPlonkProverParam, HyperPlonkVerifierParam},
        PlonkishCircuitInfo,
    },
    pcs::PolynomialCommitmentScheme,
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{div_ceil, steps, PrimeField},
        chain,
        expression::{Expression, Query, Rotation},
        Itertools,
    },
    Error,
};
use std::{array, borrow::Cow, mem};

pub(crate) fn batch_size<F: PrimeField>(circuit_info: &PlonkishCircuitInfo<F>) -> usize {
    let num_lookups = circuit_info.lookups.len();
    let num_permutation_polys = circuit_info.permutation_polys().len();
    chain![
        [circuit_info.preprocess_polys.len() + circuit_info.permutation_polys().len()],
        circuit_info.num_witness_polys.clone(),
        [num_lookups],
        [num_lookups + div_ceil(num_permutation_polys, max_degree(circuit_info, None) - 1)],
    ]
    .sum()
}

#[allow(clippy::type_complexity)]
pub(crate) fn preprocess<F: PrimeField, Pcs: PolynomialCommitmentScheme<F>>(
    param: &Pcs::Param,
    circuit_info: &PlonkishCircuitInfo<F>,
    batch_commit: impl Fn(
        &Pcs::ProverParam,
        Vec<MultilinearPolynomial<F>>,
    ) -> Result<(Vec<MultilinearPolynomial<F>>, Vec<Pcs::Commitment>), Error>,
) -> Result<
    (
        HyperPlonkProverParam<F, Pcs>,
        HyperPlonkVerifierParam<F, Pcs>,
    ),
    Error,
> {
    assert!(circuit_info.is_well_formed());

    let num_vars = circuit_info.k;
    let poly_size = 1 << num_vars;
    // Batch size for the polynomial commitment scheme 
    let batch_size = batch_size(circuit_info);
    // Trim the parameters for the PCS to those necessary for the size of the circuit
    let (pcs_pp, pcs_vp) = Pcs::trim(param, poly_size, batch_size)?;
    // Compute preprocesses comms
    let preprocess_polys = circuit_info
        .preprocess_polys
        .iter()
        .cloned()
        .map(MultilinearPolynomial::new)
        .collect_vec();
    // Batch commit to all pre-processing polynomials - i.e. fixed colummns/ selector columns
    let (preprocess_polys, preprocess_comms) = batch_commit(&pcs_pp, preprocess_polys)?;

    // Compute permutation polys and comms
    let permutation_polys = permutation_polys(
        num_vars,
        &circuit_info.permutation_polys(),
        &circuit_info.permutations,
    );
    let (permutation_polys, permutation_comms) = batch_commit(&pcs_pp, permutation_polys)?;

    // Compose an expression for all the constraints
    let (num_permutation_z_polys, expression, max_degree) = compose(circuit_info);
    // Setup parameters for verifier and prover
    let vp = HyperPlonkVerifierParam {
        pcs: pcs_vp,
        num_instances: circuit_info.num_instances.clone(),
        num_witness_polys: circuit_info.num_witness_polys.clone(),
        num_challenges: circuit_info.num_challenges.clone(),
        num_lookups: circuit_info.lookups.len(),
        num_permutation_z_polys,
        num_vars,
        expression: expression.clone(),
        preprocess_comms: preprocess_comms.clone(),
        permutation_comms: circuit_info
            .permutation_polys()
            .into_iter()
            .zip(permutation_comms.clone())
            .collect(),
    };
    let pp = HyperPlonkProverParam {
        max_degree,
        pcs: pcs_pp,
        num_instances: circuit_info.num_instances.clone(),
        num_witness_polys: circuit_info.num_witness_polys.clone(),
        num_challenges: circuit_info.num_challenges.clone(),
        lookups: circuit_info.lookups.clone(),
        num_permutation_z_polys,
        num_vars,
        expression,
        preprocess_polys,
        preprocess_comms,
        permutation_polys: circuit_info
            .permutation_polys()
            .into_iter()
            .zip(permutation_polys)
            .collect(),
        permutation_comms,
    };
    Ok((pp, vp))
}

// compose all constraints
pub(crate) fn compose<F: PrimeField>(
    circuit_info: &PlonkishCircuitInfo<F>,
) -> (usize, Expression<F>, usize) {
    //total number of challenges
    let challenge_offset = circuit_info.num_challenges.iter().sum::<usize>();
    // Generates three extra challenges beta, gamma, alpha
    let [beta, gamma, alpha] =
        &array::from_fn(|idx| Expression::<F>::Challenge(challenge_offset + idx));

    // lookup_zero_checks are the sumcheck constraints in the logup GKR protocol
    let (lookup_constraints, lookup_zero_checks) = lookup_constraints(circuit_info, beta, gamma);

    let max_degree = max_degree(circuit_info, Some(&lookup_constraints));
    // Generate constraints for the permuation argument
    let (num_permutation_z_polys, permutation_constraints) = permutation_constraints(
        circuit_info,
        max_degree,
        beta,
        gamma,
        2 * circuit_info.lookups.len(),
    );

    let expression = {
        let constraints = chain![
            // constraints from halo2 frontend , i.e. custom gates
            circuit_info.constraints.iter(),
            // constraints from lookup argument
            lookup_constraints.iter(),
            // constraints from permutation argument
            permutation_constraints.iter(),
        ]
        .collect_vec();
        let eq = Expression::eq_xy(0);
        let zero_check_on_every_row = Expression::distribute_powers(constraints, alpha) * eq;
        Expression::distribute_powers(
            chain![lookup_zero_checks.iter(), [&zero_check_on_every_row]],
            alpha,
        )
    };

    (num_permutation_z_polys, expression, max_degree)
}

pub(super) fn max_degree<F: PrimeField>(
    circuit_info: &PlonkishCircuitInfo<F>,
    lookup_constraints: Option<&[Expression<F>]>,
) -> usize {
    let lookup_constraints = lookup_constraints.map(Cow::Borrowed).unwrap_or_else(|| {
        let dummy_challenge = Expression::zero();
        Cow::Owned(self::lookup_constraints(circuit_info, &dummy_challenge, &dummy_challenge).0)
    });
    chain![
        circuit_info.constraints.iter().map(Expression::degree),
        lookup_constraints.iter().map(Expression::degree),
        circuit_info.max_degree,
        [2],
    ]
    .max()
    .unwrap()
}

//generate lookup constraints using logup GKR 
pub(super) fn lookup_constraints<F: PrimeField>(
    circuit_info: &PlonkishCircuitInfo<F>,
    beta: &Expression<F>,
    gamma: &Expression<F>,
) -> (Vec<Expression<F>>, Vec<Expression<F>>) {
    // Define indices where m and h polynomials begin in the trace
    let m_offset = circuit_info.num_poly() + circuit_info.permutation_polys().len();
    let h_offset = m_offset + circuit_info.lookups.len();
    let constraints = circuit_info
        .lookups
        .iter()
        .zip(m_offset..)
        .zip(h_offset..)
        .flat_map(|((lookup, m), h)| {
            // make m and h into polynomials, these are created during proving 
            let [m, h] = &[m, h]
                .map(|poly| Query::new(poly, Rotation::cur()))
                .map(Expression::<F>::Polynomial);
            // separate the input and tables from the lookup
            let (inputs, tables) = lookup
                .iter()
                .map(|(input, table)| (input, table))
                .unzip::<_, _, Vec<_>, Vec<_>>();
            // Returns a distributed power expression for the input and table, with base beta, i.e.  inputs[0] + \beta inputs[1] + \beta^2 inputs[2] + ...
            let input = &Expression::distribute_powers(inputs, beta);
            let table = &Expression::distribute_powers(tables, beta);
            // h[i] = (gamma + input[i])^-1 - m[i] * (gamma + table[i])^-1
            [h * (input + gamma) * (table + gamma) - (table + gamma) + m * (input + gamma)]
        })
        .collect_vec();
    // Every expression that must be proved in the sum check argument 
    let sum_check = (h_offset..)
        .take(circuit_info.lookups.len())
        .map(|h| Query::new(h, Rotation::cur()).into())
        .collect_vec();
    (constraints, sum_check)
}

// create constraints for the permutation argument
pub(crate) fn permutation_constraints<F: PrimeField>(
    circuit_info: &PlonkishCircuitInfo<F>,
    max_degree: usize,
    beta: &Expression<F>,
    gamma: &Expression<F>,
    num_builtin_witness_polys: usize,
) -> (usize, Vec<Expression<F>>) {
    // get index of all columns used in the permutation
    let permutation_polys = circuit_info.permutation_polys();
    let chunk_size = max_degree - 1;
    // If there are more columns than max degree split into chunks, num_chunks corresponds to b in halo2 gitbook
    let num_chunks = div_ceil(permutation_polys.len(), chunk_size);
    // The offset is set to the total number of instance columns in the circuit
    let permutation_offset = circuit_info.num_poly();
    let z_offset = permutation_offset + permutation_polys.len() + num_builtin_witness_polys;
    // Represent all columns in permutation argument  with polynomials 
    let polys = permutation_polys
        .iter()
        .map(|idx| Expression::Polynomial(Query::new(*idx, Rotation::cur())))
        .collect_vec();
    //ids_i(X) = i 2^k + X
    let ids = (0..polys.len())
        .map(|idx| {
            let offset = F::from((idx << circuit_info.k) as u64);
            Expression::Constant(offset) + Expression::identity()
        })
        .collect_vec();
    // Create the polynomials to represent the permutation columns
    let permutations = (permutation_offset..)
        .map(|idx| Expression::Polynomial(Query::new(idx, Rotation::cur())))
        .take(permutation_polys.len())
        .collect_vec();
    // Represents Z polynomials from the permutation argument
    let zs = (z_offset..)
        .map(|idx| Expression::Polynomial(Query::new(idx, Rotation::cur())))
        .take(num_chunks)
        .collect_vec();
    // Z_0(shift(X))
    let z_0_next = Expression::<F>::Polynomial(Query::new(z_offset, Rotation::next()));
    let l_0 = &Expression::<F>::lagrange(0);
    let one = &Expression::one();
    // Create the constraints for the permutation argument 
    // The contraints here are the like those from the halo2 gitbook but the matrix Z_0 Z_1 ... Z_{b-1}  is transposed
    let constraints = chain![
        zs.first().map(|z_0| l_0 * (z_0 - one)),
        polys
            //iterating over b elements which are vectors of length m 
            .chunks(chunk_size)
            .zip(ids.chunks(chunk_size))
            .zip(permutations.chunks(chunk_size))
            .zip(zs.iter())
            .zip(zs.iter().skip(1).chain([&z_0_next]))
            // z_a prod_{am)}^{(a+1)m-1}(poly_i + beta * id_i + gamma) - z_{a+1} prod_{am)}^{(a+1)m-1}(poly_i + beta * permutation_i + gamma)
            // z_{b-1} prod_{(b-1)m)}^{bm-1}(poly_{b-1} + beta * id_{b-1} + gamma) - z_0(shift(X)) prod_{(b-1)m)}^{bm-1}(poly_{b-1} + beta * permutation_{b-1} + gamma)
            .map(|((((polys, ids), permutations), z_lhs), z_rhs)| {
                z_lhs
                    * polys
                        .iter()
                        .zip(ids)
                        .map(|(poly, id)| poly + beta * id + gamma)
                        .product::<Expression<_>>()
                    - z_rhs
                        * polys
                            .iter()
                            .zip(permutations)
                            .map(|(poly, permutation)| poly + beta * permutation + gamma)
                            .product::<Expression<_>>()
            }),
    ]
    .collect();
    (num_chunks, constraints)
}

// Generate multi-linear permutation polynomials for permutation argument 
pub(crate) fn permutation_polys<F: PrimeField>(
    num_vars: usize,
    permutation_polys: &[usize],
    cycles: &[Vec<(usize, usize)>],
) -> Vec<MultilinearPolynomial<F>> {
    // The index of an element in permutation_polys
    let poly_index = {
        let mut poly_index = vec![0; permutation_polys.last().map(|poly| 1 + poly).unwrap_or(0)];
        for (idx, poly) in permutation_polys.iter().enumerate() {
            poly_index[*poly] = idx;
        }
        poly_index
    };
    // permutations will be the matrix defining all permutation polynomials. As we start with the identity permutation, all entries have value of the index within the matrix. 
    let mut permutations = (0..permutation_polys.len() as u64)
        .map(|idx| {
            steps(F::from(idx << num_vars))
                .take(1 << num_vars)
                .collect_vec()
        })
        .collect_vec();
    // For each cycle we update the permutation matrix. For each entry in the matrix, we have the location of the next element in the cycle.
    for cycle in cycles.iter() {
        let (i0, j0) = cycle[0];
        let mut last = permutations[poly_index[i0]][j0];
        for &(i, j) in cycle.iter().cycle().skip(1).take(cycle.len()) {
            mem::swap(&mut permutations[poly_index[i]][j], &mut last);
        }
    }
    // We generate a multilinear polynomial from each column of the permutation matrix. 
    permutations
        .into_iter()
        .map(MultilinearPolynomial::new)
        .collect()
}

#[cfg(test)]
pub(crate) mod test {
    use crate::{
        backend::hyperplonk::util::{vanilla_plonk_expression, vanilla_plonk_w_lookup_expression},
        util::expression::{Expression, Query, Rotation},
    };
    use halo2_curves::bn256::Fr;
    use std::array;

    #[test]
    fn compose_vanilla_plonk() {
        let num_vars = 3;
        let expression = vanilla_plonk_expression(num_vars);
        assert_eq!(expression, {
            let [pi, q_l, q_r, q_m, q_o, q_c, w_l, w_r, w_o, s_1, s_2, s_3] =
                &array::from_fn(|poly| Query::new(poly, Rotation::cur()))
                    .map(Expression::Polynomial);
            let [z, z_next] = &[
                Query::new(12, Rotation::cur()),
                Query::new(12, Rotation::next()),
            ]
            .map(Expression::Polynomial);
            let [beta, gamma, alpha] = &array::from_fn(Expression::<Fr>::Challenge);
            let [id_1, id_2, id_3] = array::from_fn(|idx| {
                Expression::Constant(Fr::from((idx << num_vars) as u64)) + Expression::identity()
            });
            let l_0 = Expression::<Fr>::lagrange(0);
            let one = Expression::one();
            let constraints = {
                vec![
                    q_l * w_l + q_r * w_r + q_m * w_l * w_r + q_o * w_o + q_c + pi,
                    l_0 * (z - one),
                    (z * ((w_l + beta * id_1 + gamma)
                        * (w_r + beta * id_2 + gamma)
                        * (w_o + beta * id_3 + gamma)))
                        - (z_next
                            * ((w_l + beta * s_1 + gamma)
                                * (w_r + beta * s_2 + gamma)
                                * (w_o + beta * s_3 + gamma))),
                ]
            };
            let eq = Expression::eq_xy(0);
            Expression::distribute_powers(&constraints, alpha) * eq
        });
    }

    #[test]
    fn compose_vanilla_plonk_w_lookup() {
        let num_vars = 3;
        let expression = vanilla_plonk_w_lookup_expression(num_vars);
        assert_eq!(expression, {
            let [pi, q_l, q_r, q_m, q_o, q_c, q_lookup, t_l, t_r, t_o, w_l, w_r, w_o, s_1, s_2, s_3] =
                &array::from_fn(|poly| Query::new(poly, Rotation::cur()))
                    .map(Expression::Polynomial);
            let [lookup_m, lookup_h] = &[
                Query::new(16, Rotation::cur()),
                Query::new(17, Rotation::cur()),
            ]
            .map(Expression::<Fr>::Polynomial);
            let [perm_z, perm_z_next] = &[
                Query::new(18, Rotation::cur()),
                Query::new(18, Rotation::next()),
            ]
            .map(Expression::Polynomial);
            let [beta, gamma, alpha] = &array::from_fn(Expression::<Fr>::Challenge);
            let [id_1, id_2, id_3] = array::from_fn(|idx| {
                Expression::Constant(Fr::from((idx << num_vars) as u64)) + Expression::identity()
            });
            let l_0 = &Expression::<Fr>::lagrange(0);
            let one = &Expression::one();
            let lookup_input =
                &Expression::distribute_powers(&[w_l, w_r, w_o].map(|w| q_lookup * w), beta);
            let lookup_table = &Expression::distribute_powers([t_l, t_r, t_o], beta);
            let constraints = {
                vec![
                    q_l * w_l + q_r * w_r + q_m * w_l * w_r + q_o * w_o + q_c + pi,
                    lookup_h * (lookup_input + gamma) * (lookup_table + gamma)
                        - (lookup_table + gamma)
                        + lookup_m * (lookup_input + gamma),
                    l_0 * (perm_z - one),
                    (perm_z
                        * ((w_l + beta * id_1 + gamma)
                            * (w_r + beta * id_2 + gamma)
                            * (w_o + beta * id_3 + gamma)))
                        - (perm_z_next
                            * ((w_l + beta * s_1 + gamma)
                                * (w_r + beta * s_2 + gamma)
                                * (w_o + beta * s_3 + gamma))),
                ]
            };
            let eq = Expression::eq_xy(0);
            let zero_check_on_every_row = Expression::distribute_powers(&constraints, alpha) * eq;
            let lookup_zero_check = lookup_h;
            Expression::distribute_powers([lookup_zero_check, &zero_check_on_every_row], alpha)
        });
    }
}
