use crate::{
    backend::{
        hyperplonk::{
            preprocessor::{batch_size, preprocess},
            prover::{
                instance_polys, lookup_compressed_polys, lookup_h_polys, lookup_m_polys,
                permutation_z_polys, prove_zero_check,
            },
            verifier::verify_zero_check,
        },
        PlonkishBackend, PlonkishCircuit, PlonkishCircuitInfo, WitnessEncoding,
    },
    pcs::PolynomialCommitmentScheme,
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{powers, PrimeField},
        chain, end_timer,
        expression::{
            rotate::{BinaryField, Rotatable},
            Expression,
        },
        start_timer,
        transcript::{TranscriptRead, TranscriptWrite},
        Deserialize, DeserializeOwned, Itertools, Serialize,
    },
    Error,
};
use rand::RngCore;
use std::{fmt::Debug, hash::Hash, iter, marker::PhantomData};

pub(crate) mod preprocessor;
pub(crate) mod prover;
pub(crate) mod verifier;

#[cfg(any(test, feature = "benchmark"))]
pub mod util;

#[derive(Clone, Debug)]
pub struct HyperPlonk<Pcs>(PhantomData<Pcs>);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HyperPlonkProverParam<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    pub(crate) pcs: Pcs::ProverParam,
    pub(crate) num_instances: Vec<usize>,
    pub(crate) num_witness_polys: Vec<usize>,
    pub(crate) num_challenges: Vec<usize>,
    pub(crate) lookups: Vec<Vec<(Expression<F>, Expression<F>)>>,
    pub(crate) num_permutation_z_polys: usize,
    pub(crate) num_vars: usize,
    pub(crate) expression: Expression<F>,
    pub(crate) preprocess_polys: Vec<MultilinearPolynomial<F>>,
    pub(crate) preprocess_comms: Vec<Pcs::Commitment>,
    pub(crate) permutation_polys: Vec<(usize, MultilinearPolynomial<F>)>,
    pub(crate) permutation_comms: Vec<Pcs::Commitment>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HyperPlonkVerifierParam<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    pub(crate) pcs: Pcs::VerifierParam,
    pub(crate) num_instances: Vec<usize>,
    pub(crate) num_witness_polys: Vec<usize>,
    pub(crate) num_challenges: Vec<usize>,
    pub(crate) num_lookups: usize,
    pub(crate) num_permutation_z_polys: usize,
    pub(crate) num_vars: usize,
    // Expression for the constraint system
    pub(crate) expression: Expression<F>,
    pub(crate) preprocess_comms: Vec<Pcs::Commitment>,
    pub(crate) permutation_comms: Vec<(usize, Pcs::Commitment)>,
}

impl<F, Pcs> PlonkishBackend<F> for HyperPlonk<Pcs>
where
    F: PrimeField + Hash + Serialize + DeserializeOwned,
    Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
{
    type Pcs = Pcs;
    type ProverParam = HyperPlonkProverParam<F, Pcs>;
    type VerifierParam = HyperPlonkVerifierParam<F, Pcs>;

    //TO DO - for KZG we want to use a trusted setup from a ceremony as currently generated in nightfish/ nightfall
    fn setup(
        circuit_info: &PlonkishCircuitInfo<F>,
        rng: impl RngCore,
    ) -> Result<Pcs::Param, Error> {
        assert!(circuit_info.is_well_formed());

        let num_vars = circuit_info.k;
        // number of variables in the multilinear polynomial
        let poly_size = 1 << num_vars;
        // set batch size for polynomial commitment scheme
        let batch_size = batch_size(circuit_info);
        Pcs::setup(poly_size, batch_size, rng)
    }

    fn preprocess(
        param: &Pcs::Param,
        circuit_info: &PlonkishCircuitInfo<F>,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error> {
        preprocess(param, circuit_info, |pp, polys| {
            let comms = Pcs::batch_commit(pp, &polys)?;
            Ok((polys, comms))
        })
    }

    fn prove(
        pp: &Self::ProverParam,
        circuit: &impl PlonkishCircuit<F>,
        transcript: &mut impl TranscriptWrite<Pcs::CommitmentChunk, F>,
        _: impl RngCore,
    ) -> Result<(), Error> {
        let instance_polys = {
            let instances = circuit.instances();
            //Check there is the correct amount of instances and write them to the transcript
            for (num_instances, instances) in pp.num_instances.iter().zip_eq(instances) {
                assert_eq!(instances.len(), *num_instances);
                for instance in instances.iter() {
                    transcript.common_field_element(instance)?;
                }
            }
            // Create multi-linear polynomials from the instance columns
            instance_polys::<_, BinaryField>(pp.num_vars, instances)
        };

        // Round 0..n

        let mut witness_polys = Vec::with_capacity(pp.num_witness_polys.iter().sum());
        let mut witness_comms = Vec::with_capacity(witness_polys.len());
        let mut challenges = Vec::with_capacity(pp.num_challenges.iter().sum::<usize>() + 4);
        // For each round, generate multi-linear polynomials from witness columns and commit
        for (round, (num_witness_polys, num_challenges)) in pp
            .num_witness_polys
            .iter()
            .zip_eq(pp.num_challenges.iter())
            .enumerate()
        {
            let timer = start_timer(|| format!("witness_collector-{round}"));
            let polys = circuit
                .synthesize(round, &challenges)?
                .into_iter()
                .map(MultilinearPolynomial::new)
                .collect_vec();
            assert_eq!(polys.len(), *num_witness_polys);
            end_timer(timer);

            witness_comms.extend(Pcs::batch_commit_and_write(&pp.pcs, &polys, transcript)?);
            witness_polys.extend(polys);
            challenges.extend(transcript.squeeze_challenges(*num_challenges));
        }
        let polys = chain![&instance_polys, &pp.preprocess_polys, &witness_polys].collect_vec();

        // Round n

        // beta is used to compress the polynomials in the lookup argument
        let beta = transcript.squeeze_challenge();

        // Generate a compressed multilinear polynomial for each lookup in the vector of lookups
        let timer = start_timer(|| format!("lookup_compressed_polys-{}", pp.lookups.len()));
        let lookup_compressed_polys = {
            let max_lookup_width = pp.lookups.iter().map(Vec::len).max().unwrap_or_default();
            let betas = powers(beta).take(max_lookup_width).collect_vec();
            lookup_compressed_polys::<_, BinaryField>(&pp.lookups, &polys, &challenges, &betas)
        };
        end_timer(timer);

        // m and h are the polynomials generated as part of the logup GKR protocol
        // if the lookups are f (input), t (table), m[i] is |j \in 2^{numvars} s.t f[j] = t[i] |, i.e. the number of times t[i] appears in f
        let timer = start_timer(|| format!("lookup_m_polys-{}", pp.lookups.len()));
        let lookup_m_polys = lookup_m_polys(&lookup_compressed_polys)?;
        end_timer(timer);

        let lookup_m_comms = Pcs::batch_commit_and_write(&pp.pcs, &lookup_m_polys, transcript)?;

        // Round n+1

        let gamma = transcript.squeeze_challenge();

        //h = (gamma + input)^-1 - m * (gamma + table)^-1
        let timer = start_timer(|| format!("lookup_h_polys-{}", pp.lookups.len()));
        let lookup_h_polys = lookup_h_polys(&lookup_compressed_polys, &lookup_m_polys, &gamma);
        end_timer(timer);

        let timer = start_timer(|| format!("permutation_z_polys-{}", pp.permutation_polys.len()));
        let permutation_z_polys = permutation_z_polys::<_, BinaryField>(
            pp.num_permutation_z_polys,
            &pp.permutation_polys,
            &polys,
            &beta,
            &gamma,
        );
        end_timer(timer);

        // Commit to h polynomiald and permutation z polynomials
        let lookup_h_permutation_z_polys =
            chain![lookup_h_polys.iter(), permutation_z_polys.iter()].collect_vec();
        let lookup_h_permutation_z_comms =
            Pcs::batch_commit_and_write(&pp.pcs, lookup_h_permutation_z_polys.clone(), transcript)?;

        // Round n+2

        let alpha = transcript.squeeze_challenge();
        let y = transcript.squeeze_challenges(pp.num_vars);

        // All of the polynomials in the trace committed to in the transcript
        let polys = chain![
            polys,
            pp.permutation_polys.iter().map(|(_, poly)| poly),
            lookup_m_polys.iter(),
            lookup_h_permutation_z_polys,
        ]
        .collect_vec();
        challenges.extend([beta, gamma, alpha]);
        // Prove the zero check is satisfied for the expression wrt the polynomials
        let (points, evals) = prove_zero_check(
            pp.num_instances.len(),
            &pp.expression,
            &polys,
            challenges,
            y,
            transcript,
        )?;

        // PCS open

        let dummy_comm = Pcs::Commitment::default();
        let comms = chain![
            iter::repeat(&dummy_comm).take(pp.num_instances.len()),
            &pp.preprocess_comms,
            &witness_comms,
            &pp.permutation_comms,
            &lookup_m_comms,
            &lookup_h_permutation_z_comms,
        ]
        .collect_vec();
        let timer = start_timer(|| format!("pcs_batch_open-{}", evals.len()));
        // Open all polynomials at the points from the zero check and give the opening proofs
        Pcs::batch_open(&pp.pcs, polys, comms, &points, &evals, transcript)?;
        end_timer(timer);
        // Proof is saved in transcript
        Ok(())
    }

    fn verify(
        vp: &Self::VerifierParam,
        instances: &[Vec<F>],
        transcript: &mut impl TranscriptRead<Pcs::CommitmentChunk, F>,
        _: impl RngCore,
    ) -> Result<(), Error> {
        //Check there is the correct amount of instances and write them to the transcript
        for (num_instances, instances) in vp.num_instances.iter().zip_eq(instances) {
            assert_eq!(instances.len(), *num_instances);
            for instance in instances.iter() {
                transcript.common_field_element(instance)?;
            }
        }

        // Round 0..n

        // For each round, read the witness commitments from the transcript and generate the challenges
        let mut witness_comms = Vec::with_capacity(vp.num_witness_polys.iter().sum());
        let mut challenges = Vec::with_capacity(vp.num_challenges.iter().sum::<usize>() + 4);
        for (num_polys, num_challenges) in
            vp.num_witness_polys.iter().zip_eq(vp.num_challenges.iter())
        {
            witness_comms.extend(Pcs::read_commitments(&vp.pcs, *num_polys, transcript)?);
            challenges.extend(transcript.squeeze_challenges(*num_challenges));
        }

        // Round n

        let beta = transcript.squeeze_challenge();

        // Read the commitments to the m polynomials from the lookup arguments
        let lookup_m_comms = Pcs::read_commitments(&vp.pcs, vp.num_lookups, transcript)?;

        // Round n+1

        let gamma = transcript.squeeze_challenge();

        // Read the commitments to the h polynomials and permutation z polynomials
        let lookup_h_permutation_z_comms = Pcs::read_commitments(
            &vp.pcs,
            vp.num_lookups + vp.num_permutation_z_polys,
            transcript,
        )?;

        // Round n+2

        let alpha = transcript.squeeze_challenge();
        let y = transcript.squeeze_challenges(vp.num_vars);

        challenges.extend([beta, gamma, alpha]);
        // Verify the zero check for the constraints defined in the expression
        let (points, evals) = verify_zero_check(
            vp.num_vars,
            &vp.expression,
            instances,
            &challenges,
            &y,
            transcript,
        )?;

        // PCS verify

        let dummy_comm = Pcs::Commitment::default();
        let comms = chain![
            iter::repeat(&dummy_comm).take(vp.num_instances.len()),
            &vp.preprocess_comms,
            &witness_comms,
            vp.permutation_comms.iter().map(|(_, comm)| comm),
            &lookup_m_comms,
            &lookup_h_permutation_z_comms,
        ]
        .collect_vec();
        // Verify the opening proofs for the polynomials commitments
        Pcs::batch_verify(&vp.pcs, comms, &points, &evals, transcript)?;

        Ok(())
    }
}

impl<Pcs> WitnessEncoding for HyperPlonk<Pcs> {
    fn row_mapping(k: usize) -> Vec<usize> {
        BinaryField::new(k).usable_indices()
    }
}

#[cfg(test)]
mod test {
    use crate::{
        backend::{
            hyperplonk::{
                util::{rand_vanilla_plonk_circuit, rand_vanilla_plonk_w_lookup_circuit},
                HyperPlonk,
            },
            test::run_plonkish_backend,
        },
        pcs::{
            multilinear::{Gemini, Zeromorph},
            univariate::{UnivariateIpa, UnivariateKzg},
        },
        util::{
            expression::rotate::BinaryField, test::seeded_std_rng, transcript::Keccak256Transcript,
        },
    };
    use halo2_curves::{bn256::Bn256, grumpkin};

    macro_rules! tests {
        ($suffix:ident, $pcs:ty, $num_vars_range:expr) => {
            paste::paste! {
                #[test]
                fn [<vanilla_plonk_w_ $suffix>]() {
                    run_plonkish_backend::<_, HyperPlonk<$pcs>, Keccak256Transcript<_>, _>($num_vars_range, |num_vars| {
                        rand_vanilla_plonk_circuit::<_, BinaryField>(num_vars, seeded_std_rng(), seeded_std_rng())
                    });
                }

                #[test]
                fn [<vanilla_plonk_w_lookup_w_ $suffix>]() {
                    run_plonkish_backend::<_, HyperPlonk<$pcs>, Keccak256Transcript<_>, _>($num_vars_range, |num_vars| {
                        rand_vanilla_plonk_w_lookup_circuit::<_, BinaryField>(num_vars, seeded_std_rng(), seeded_std_rng())
                    });
                }
            }
        };
        ($suffix:ident, $pcs:ty) => {
            tests!($suffix, $pcs, 2..16);
        };
    }

    //tests!(brakedown, MultilinearBrakedown<bn256::Fr, Keccak256, BrakedownSpec6>);
    //tests!(hyrax, MultilinearHyrax<grumpkin::G1Affine>, 5..16);
    //tests!(ipa, MultilinearIpa<grumpkin::G1Affine>);
    //tests!(kzg, MultilinearKzg<Bn256>);
    tests!(gemini_kzg, Gemini<UnivariateKzg<Bn256>>);
    tests!(zeromorph_kzg, Zeromorph<UnivariateKzg<Bn256>>);
    tests!(gemini_ipa, Gemini<UnivariateIpa<grumpkin::G1Affine>>);
    //tests!(zeromorph_ipa, Zeromorph<UnivariateIpa<grumpkin::G1Affine>>);
}
