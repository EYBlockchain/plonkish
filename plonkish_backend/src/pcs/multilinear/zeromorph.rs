// See https://eprint.iacr.org/2023/917.pdf for details of the zeromorph protocol
use crate::{
    pcs::{
        multilinear::{additive, quotients},
        univariate::{
            err_too_large_degree, UnivariateIpa, UnivariateIpaCommitment, UnivariateKzg,
            UnivariateKzgProverParam, UnivariateKzgVerifierParam,
        },
        Evaluation, Point, PolynomialCommitmentScheme,
    },
    poly::{multilinear::MultilinearPolynomial, univariate::UnivariatePolynomial},
    util::{
        arithmetic::{
            powers, squares, variable_base_msm, BatchInvert, Curve, Field, MultiMillerLoop,
        },
        chain, izip,
        parallel::parallelize,
        transcript::{TranscriptRead, TranscriptWrite},
        Deserialize, DeserializeOwned, Itertools, Serialize,
    },
    Error,
};
use halo2_curves::CurveAffine;
use rand::RngCore;
use std::marker::PhantomData;

#[derive(Clone, Debug)]
pub struct Zeromorph<Pcs>(PhantomData<Pcs>);

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "M::G1Affine: Serialize",
    deserialize = "M::G1Affine: DeserializeOwned",
))]
pub struct ZeromorphKzgProverParam<M: MultiMillerLoop> {
    commit_pp: UnivariateKzgProverParam<M>,
    open_pp: UnivariateKzgProverParam<M>,
}

impl<M: MultiMillerLoop> ZeromorphKzgProverParam<M> {
    pub fn degree(&self) -> usize {
        self.commit_pp.degree()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "M::G1Affine: Serialize, M::G2Affine: Serialize",
    deserialize = "M::G1Affine: DeserializeOwned, M::G2Affine: DeserializeOwned",
))]
pub struct ZeromorphKzgVerifierParam<M: MultiMillerLoop> {
    vp: UnivariateKzgVerifierParam<M>,
    s_offset_g2: M::G2Affine,
}

impl<M: MultiMillerLoop> ZeromorphKzgVerifierParam<M> {
    pub fn g1(&self) -> M::G1Affine {
        self.vp.g1()
    }

    pub fn g2(&self) -> M::G2Affine {
        self.vp.g2()
    }

    pub fn s_g2(&self) -> M::G2Affine {
        self.vp.s_g2()
    }
}

impl<M> PolynomialCommitmentScheme<M::Scalar> for Zeromorph<UnivariateKzg<M>>
where
    M: MultiMillerLoop,
    M::Scalar: Serialize + DeserializeOwned,
    M::G1Affine: Serialize + DeserializeOwned,
    M::G2Affine: Serialize + DeserializeOwned,
{
    type Param = <UnivariateKzg<M> as PolynomialCommitmentScheme<M::Scalar>>::Param;
    type ProverParam = ZeromorphKzgProverParam<M>;
    type VerifierParam = ZeromorphKzgVerifierParam<M>;
    type Polynomial = MultilinearPolynomial<M::Scalar>;
    type Commitment = <UnivariateKzg<M> as PolynomialCommitmentScheme<M::Scalar>>::Commitment;
    type CommitmentChunk =
        <UnivariateKzg<M> as PolynomialCommitmentScheme<M::Scalar>>::CommitmentChunk;

    fn setup(poly_size: usize, batch_size: usize, rng: impl RngCore) -> Result<Self::Param, Error> {
        assert!(poly_size.is_power_of_two());

        UnivariateKzg::<M>::setup(poly_size, batch_size, rng)
    }

    fn trim(
        param: &Self::Param,
        poly_size: usize,
        batch_size: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error> {
        assert!(poly_size.is_power_of_two());

        let (commit_pp, vp) = UnivariateKzg::<M>::trim(param, poly_size, batch_size)?;
        let offset = param.monomial_g1().len() - poly_size;
        let open_pp = {
            let monomial_g1 = param.monomial_g1()[offset..].to_vec();
            UnivariateKzgProverParam::new(poly_size.ilog2() as usize, monomial_g1, Vec::new())
        };
        let s_offset_g2 = param.powers_of_s_g2()[offset];

        Ok((
            ZeromorphKzgProverParam { commit_pp, open_pp },
            ZeromorphKzgVerifierParam { vp, s_offset_g2 },
        ))
    }

    fn commit(pp: &Self::ProverParam, poly: &Self::Polynomial) -> Result<Self::Commitment, Error> {
        if pp.degree() + 1 < poly.evals().len() {
            let got = poly.evals().len() - 1;
            return Err(err_too_large_degree("commit", pp.degree(), got));
        }

        Ok(UnivariateKzg::commit_monomial(&pp.commit_pp, poly.evals()))
    }

    fn batch_commit<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
    ) -> Result<Vec<Self::Commitment>, Error> {
        polys
            .into_iter()
            .map(|poly| Self::commit(pp, poly))
            .collect()
    }

    fn open(
        pp: &Self::ProverParam,
        poly: &Self::Polynomial,
        comm: &Self::Commitment,
        point: &Point<M::Scalar, Self::Polynomial>,
        eval: &M::Scalar,
        transcript: &mut impl TranscriptWrite<Self::CommitmentChunk, M::Scalar>,
    ) -> Result<(), Error> {
        let num_vars = poly.num_vars();
        if pp.degree() + 1 < poly.evals().len() {
            let got = poly.evals().len() - 1;
            return Err(err_too_large_degree("open", pp.degree(), got));
        }

        if cfg!(feature = "sanity-check") {
            assert_eq!(Self::commit(pp, poly).unwrap().0, comm.0);
            assert_eq!(poly.evaluate(point), *eval);
        }

        // We first construct the `U_i(q_i)`s and the `v` from the zeromorph paper.
        let (quotients, remainder) =
            quotients(poly, point, |_, q| UnivariatePolynomial::monomial(q));
        UnivariateKzg::batch_commit_and_write(&pp.commit_pp, &quotients, transcript)?;

        if cfg!(feature = "sanity-check") {
            assert_eq!(&remainder, eval);
        }

        let y = transcript.squeeze_challenge();

        // Here `q_hat = sum_{i=0}^{num_vars - 1} y^i X^{2^num_vars - 2^i} U_i(q_i)`.
        // The powers of `X` are used to degree check all the `U_i(q_i)`s.
        let q_hat = {
            let mut q_hat = vec![M::Scalar::ZERO; 1 << num_vars];
            for (idx, (power_of_y, q)) in izip!(powers(y), &quotients).enumerate() {
                let offset = (1 << num_vars) - (1 << idx);
                parallelize(&mut q_hat[offset..], |(q_hat, start)| {
                    izip!(q_hat, q.iter().skip(start))
                        .for_each(|(q_hat, q)| *q_hat += power_of_y * q)
                });
            }
            UnivariatePolynomial::monomial(q_hat)
        };
        UnivariateKzg::commit_and_write(&pp.commit_pp, &q_hat, transcript)?;

        let x = transcript.squeeze_challenge();
        let z = transcript.squeeze_challenge();

        let (eval_scalar, q_scalars) = eval_and_quotient_scalars(y, x, z, point);

        let mut f = UnivariatePolynomial::monomial(poly.evals().to_vec());
        f *= &z;
        f += &q_hat;
        f[0] += eval_scalar * eval;
        izip!(&quotients, &q_scalars).for_each(|(q, scalar)| f += (scalar, q));

        let comm = if cfg!(feature = "sanity-check") {
            assert_eq!(f.evaluate(&x), M::Scalar::ZERO);

            UnivariateKzg::commit_monomial(&pp.open_pp, f.coeffs())
        } else {
            Default::default()
        };

        UnivariateKzg::<M>::open(&pp.open_pp, &f, &comm, &x, &M::Scalar::ZERO, transcript)
    }

    fn batch_open<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
        comms: impl IntoIterator<Item = &'a Self::Commitment>,
        points: &[Point<M::Scalar, Self::Polynomial>],
        evals: &[Evaluation<M::Scalar>],
        transcript: &mut impl TranscriptWrite<Self::CommitmentChunk, M::Scalar>,
    ) -> Result<(), Error>
    where
        Self::Commitment: 'a,
    {
        let polys = polys.into_iter().collect_vec();
        let comms = comms.into_iter().collect_vec();
        let num_vars = points.first().map(|point| point.len()).unwrap_or_default();
        additive::batch_open::<_, Self>(pp, num_vars, polys, comms, points, evals, transcript)
    }

    fn read_commitments(
        vp: &Self::VerifierParam,
        num_polys: usize,
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, M::Scalar>,
    ) -> Result<Vec<Self::Commitment>, Error> {
        UnivariateKzg::read_commitments(&vp.vp, num_polys, transcript)
    }

    fn verify(
        vp: &Self::VerifierParam,
        comm: &Self::Commitment,
        point: &Point<M::Scalar, Self::Polynomial>,
        eval: &M::Scalar,
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, M::Scalar>,
    ) -> Result<(), Error> {
        let num_vars = point.len();

        let q_comms = transcript.read_commitments(num_vars)?;

        let y = transcript.squeeze_challenge();

        let q_hat_comm = transcript.read_commitment()?;

        let x = transcript.squeeze_challenge();
        let z = transcript.squeeze_challenge();

        let (eval_scalar, q_scalars) = eval_and_quotient_scalars(y, x, z, point);

        // Here we reconstruct the commitment `c` to the polynomial `f`, constructed by the prover.
        let scalars = chain![[M::Scalar::ONE, z, eval_scalar * eval], q_scalars].collect_vec();
        let bases = chain![[q_hat_comm, comm.0, vp.g1()], q_comms].collect_vec();
        let c = variable_base_msm(&scalars, &bases).into();

        let pi = transcript.read_commitment()?;

        M::pairings_product_is_identity(&[
            (&c, &(-vp.s_offset_g2).into()),
            (&pi, &(vp.s_g2() - (vp.g2() * x).into()).to_affine().into()),
        ])
        .then_some(())
        .ok_or_else(|| Error::InvalidPcsOpen("Invalid Zeromorph KZG open".to_string()))
    }

    fn batch_verify<'a>(
        vp: &Self::VerifierParam,
        comms: impl IntoIterator<Item = &'a Self::Commitment>,
        points: &[Point<M::Scalar, Self::Polynomial>],
        evals: &[Evaluation<M::Scalar>],
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, M::Scalar>,
    ) -> Result<(), Error> {
        let num_vars = points.first().map(|point| point.len()).unwrap_or_default();
        let comms = comms.into_iter().collect_vec();
        additive::batch_verify::<_, Self>(vp, num_vars, comms, points, evals, transcript)
    }
}

impl<C> PolynomialCommitmentScheme<C::Scalar> for Zeromorph<UnivariateIpa<C>>
where
    C: CurveAffine + Serialize + DeserializeOwned,
    C::ScalarExt: Serialize + DeserializeOwned,
{
    type Param = <UnivariateIpa<C> as PolynomialCommitmentScheme<C::Scalar>>::Param;
    type ProverParam = <UnivariateIpa<C> as PolynomialCommitmentScheme<C::Scalar>>::ProverParam;
    type VerifierParam = <UnivariateIpa<C> as PolynomialCommitmentScheme<C::Scalar>>::VerifierParam;
    type Polynomial = MultilinearPolynomial<C::Scalar>;
    type Commitment = <UnivariateIpa<C> as PolynomialCommitmentScheme<C::Scalar>>::Commitment;
    type CommitmentChunk =
        <UnivariateIpa<C> as PolynomialCommitmentScheme<C::Scalar>>::CommitmentChunk;

    fn setup(poly_size: usize, batch_size: usize, rng: impl RngCore) -> Result<Self::Param, Error> {
        assert!(poly_size.is_power_of_two());

        UnivariateIpa::<C>::setup(poly_size, batch_size, rng)
    }

    fn trim(
        param: &Self::Param,
        poly_size: usize,
        batch_size: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error> {
        assert!(poly_size.is_power_of_two());
        // For the moment we assume the `poly_size` is the maximum supported by the IPA PCS.
        // This ensures we don't need to do any additional degree checks
        // We may change this later.
        assert_eq!(param.monomial().len(), poly_size);

        UnivariateIpa::<C>::trim(param, poly_size, batch_size)
    }

    fn commit(pp: &Self::ProverParam, poly: &Self::Polynomial) -> Result<Self::Commitment, Error> {
        if pp.degree() + 1 < poly.evals().len() {
            let got = poly.evals().len() - 1;
            return Err(err_too_large_degree("commit", pp.degree(), got));
        }

        let bases = pp.monomial();
        Ok(UnivariateIpaCommitment(
            variable_base_msm(poly.evals(), &bases[..poly.evals().len()]).into(),
        ))
    }

    fn batch_commit<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
    ) -> Result<Vec<Self::Commitment>, Error> {
        polys
            .into_iter()
            .map(|poly| Self::commit(pp, poly))
            .collect()
    }

    fn open(
        pp: &Self::ProverParam,
        poly: &Self::Polynomial,
        comm: &Self::Commitment,
        point: &Point<C::Scalar, Self::Polynomial>,
        eval: &C::Scalar,
        transcript: &mut impl TranscriptWrite<Self::CommitmentChunk, C::Scalar>,
    ) -> Result<(), Error> {
        let num_vars = poly.num_vars();
        if pp.degree() + 1 < poly.evals().len() {
            let got = poly.evals().len() - 1;
            return Err(err_too_large_degree("open", pp.degree(), got));
        }

        if cfg!(feature = "sanity-check") {
            assert_eq!(Self::commit(pp, poly).unwrap().0, comm.0);
            assert_eq!(poly.evaluate(point), *eval);
        }

        // We first construct the `U_i(q_i)`s and the `v` from the zeromorph paper.
        let (quotients, remainder) =
            quotients(poly, point, |_, q| UnivariatePolynomial::monomial(q));
        UnivariateIpa::batch_commit_and_write(pp, &quotients, transcript)?;

        if cfg!(feature = "sanity-check") {
            assert_eq!(&remainder, eval);
        }

        let y = transcript.squeeze_challenge();

        // Here `q_hat = sum_{i=0}^{num_vars - 1} y^i X^{2^num_vars - 2^i} U_i(q_i)`.
        // The powers of `X` are used to degree check all the `U_i(q_i)`s.
        let q_hat = {
            let mut q_hat = vec![C::Scalar::ZERO; 1 << num_vars];
            for (idx, (power_of_y, q)) in izip!(powers(y), &quotients).enumerate() {
                let offset = (1 << num_vars) - (1 << idx);
                parallelize(&mut q_hat[offset..], |(q_hat, start)| {
                    izip!(q_hat, q.iter().skip(start))
                        .for_each(|(q_hat, q)| *q_hat += power_of_y * q)
                });
            }
            UnivariatePolynomial::monomial(q_hat)
        };
        UnivariateIpa::commit_and_write(pp, &q_hat, transcript)?;

        let x = transcript.squeeze_challenge();
        let z = transcript.squeeze_challenge();

        let (eval_scalar, q_scalars) = eval_and_quotient_scalars(y, x, z, point);

        let mut f = UnivariatePolynomial::monomial(poly.evals().to_vec());
        f *= &z;
        f += &q_hat;
        f[0] += eval_scalar * eval;
        izip!(&quotients, &q_scalars).for_each(|(q, scalar)| f += (scalar, q));

        let comm = if cfg!(feature = "sanity-check") {
            assert_eq!(f.evaluate(&x), C::Scalar::ZERO);

            UnivariateIpa::<C>::commit(pp, &f)?
        } else {
            Default::default()
        };

        UnivariateIpa::<C>::open(pp, &f, &comm, &x, &C::Scalar::ZERO, transcript)
    }

    fn batch_open<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
        comms: impl IntoIterator<Item = &'a Self::Commitment>,
        points: &[Point<C::Scalar, Self::Polynomial>],
        evals: &[Evaluation<C::Scalar>],
        transcript: &mut impl TranscriptWrite<Self::CommitmentChunk, C::Scalar>,
    ) -> Result<(), Error>
    where
        Self::Commitment: 'a,
    {
        let polys = polys.into_iter().collect_vec();
        let comms = comms.into_iter().collect_vec();
        let num_vars = points.first().map(|point| point.len()).unwrap_or_default();
        additive::batch_open::<_, Self>(pp, num_vars, polys, comms, points, evals, transcript)
    }

    fn read_commitments(
        vp: &Self::VerifierParam,
        num_polys: usize,
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, C::Scalar>,
    ) -> Result<Vec<Self::Commitment>, Error> {
        UnivariateIpa::read_commitments(vp, num_polys, transcript)
    }

    fn verify(
        vp: &Self::VerifierParam,
        comm: &Self::Commitment,
        point: &Point<C::Scalar, Self::Polynomial>,
        eval: &C::Scalar,
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, C::Scalar>,
    ) -> Result<(), Error> {
        let num_vars = point.len();

        let q_comms = transcript.read_commitments(num_vars)?;

        let y = transcript.squeeze_challenge();

        let q_hat_comm = transcript.read_commitment()?;

        let x = transcript.squeeze_challenge();
        let z = transcript.squeeze_challenge();

        let (eval_scalar, q_scalars) = eval_and_quotient_scalars(y, x, z, point);

        // Here we reconstruct the commitment `c` to the polynomial `f`, constructed by the prover.
        let scalars = chain![[C::Scalar::ONE, z, eval_scalar * eval], q_scalars].collect_vec();
        let bases = chain![[q_hat_comm, comm.0, vp.monomial()[0]], q_comms].collect_vec();
        let c = variable_base_msm(&scalars, &bases).into();

        UnivariateIpa::<C>::verify(
            vp,
            &UnivariateIpaCommitment::<C>(c),
            &x,
            &C::Scalar::ZERO,
            transcript,
        )
    }

    fn batch_verify<'a>(
        vp: &Self::VerifierParam,
        comms: impl IntoIterator<Item = &'a Self::Commitment>,
        points: &[Point<C::Scalar, Self::Polynomial>],
        evals: &[Evaluation<C::Scalar>],
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, C::Scalar>,
    ) -> Result<(), Error> {
        let num_vars = points.first().map(|point| point.len()).unwrap_or_default();
        let comms = comms.into_iter().collect_vec();
        additive::batch_verify::<_, Self>(vp, num_vars, comms, points, evals, transcript)
    }
}

/// returns `(-z * x^{2^num_vars} - 1)/(x - 1), q_scalars)`.
/// See below for definition of `q_scalars`.
fn eval_and_quotient_scalars<F: Field>(y: F, x: F, z: F, u: &[F]) -> (F, Vec<F>) {
    let num_vars = u.len();

    // `squares_of_x = [x, x^2, x^4, ..., x^{2^num_vars}]`
    let squares_of_x = squares(x).take(num_vars + 1).collect_vec();
    // `offsets_of_x = [x^{2^num_vars - 1}, x^{2^num_vars - 2}, x^{2^num_vars - 4}, ..., x^{2^num_vars - 2^{num_vars - 1}}]`
    let offsets_of_x = {
        let mut offsets_of_x = squares_of_x
            .iter()
            .rev()
            .skip(1)
            .scan(F::ONE, |state, power_of_x| {
                *state *= power_of_x;
                Some(*state)
            })
            .collect_vec();
        offsets_of_x.reverse();
        offsets_of_x
    };
    // `vs = [(x^{2^num_vars} - 1)/(x - 1), x^{2^num_vars} - 1)/(x^2 - 1), x^{2^num_vars} - 1)/(x^4 - 1), ..., x^{2^num_vars} - 1)/(x^{2^num_vars} - 1)]`
    let vs = {
        let v_numer = squares_of_x[num_vars] - F::ONE;
        let mut v_denoms = squares_of_x
            .iter()
            .map(|square_of_x| *square_of_x - F::ONE)
            .collect_vec();
        v_denoms.batch_invert();
        v_denoms
            .iter()
            .map(|v_denom| v_numer * v_denom)
            .collect_vec()
    };
    // i_th element of `q_scalars` is -(y^i * x^{2^n - 2^i} + z(x^{2^i}(x^{2^n - 1})/(x^{2^{i+1}} - 1) - u_i(x^{2^n - 1})/(x^{2^i} - 1)))
    let q_scalars = izip!(powers(y), offsets_of_x, squares_of_x, &vs, &vs[1..], u)
        .map(|(power_of_y, offset_of_x, square_of_x, v_i, v_j, u_i)| {
            -(power_of_y * offset_of_x + z * (square_of_x * v_j - *u_i * v_i))
        })
        .collect_vec();

    (-vs[0] * z, q_scalars)
}

#[cfg(test)]
mod test {
    use crate::{
        pcs::{
            multilinear::zeromorph::Zeromorph,
            test::{run_batch_commit_open_verify, run_commit_open_verify},
            univariate::{ipa::UnivariateIpa, UnivariateKzg},
        },
        util::transcript::Keccak256Transcript,
    };
    use halo2_curves::{bn256::Bn256, grumpkin};

    type ZeroKzgPcs = Zeromorph<UnivariateKzg<Bn256>>;
    type ZeroIpaPcs = Zeromorph<UnivariateIpa<grumpkin::G1Affine>>;

    #[test]
    fn commit_open_verify() {
        run_commit_open_verify::<_, ZeroKzgPcs, Keccak256Transcript<_>>();
        run_commit_open_verify::<_, ZeroIpaPcs, Keccak256Transcript<_>>();
    }

    #[test]
    fn batch_commit_open_verify() {
        run_batch_commit_open_verify::<_, ZeroKzgPcs, Keccak256Transcript<_>>();
        run_batch_commit_open_verify::<_, ZeroIpaPcs, Keccak256Transcript<_>>();
    }
}
