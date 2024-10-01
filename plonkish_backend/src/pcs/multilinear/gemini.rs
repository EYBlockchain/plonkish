//! Implementation of section 2.4.2 of 2022/420, with improvement ported from Aztec's Barretenberg
//! https://github.com/AztecProtocol/barretenberg/blob/master/cpp/src/barretenberg/honk/pcs/gemini/gemini.cpp.

use crate::{
    pcs::{
        multilinear::additive,
        univariate::{err_too_large_deree, UnivariateIpa, UnivariateIpaCommitment, UnivariateKzg, UnivariateKzgCommitment},
        Evaluation, Point, PolynomialCommitmentScheme,
    },
    poly::{
        multilinear::{merge_into, MultilinearPolynomial},
        univariate::UnivariatePolynomial,
    },
    util::{
        arithmetic::{squares, CurveAffine, Field, MultiMillerLoop},
        chain,
        transcript::{TranscriptRead, TranscriptWrite},
        DeserializeOwned, Itertools, Serialize,
    },
    Error,
};
use rand::RngCore;
use std::{marker::PhantomData, ops::Neg};

#[derive(Clone, Debug)]
pub struct Gemini<Pcs>(PhantomData<Pcs>);

impl<M> PolynomialCommitmentScheme<M::Scalar> for Gemini<UnivariateKzg<M>>
where
    M: MultiMillerLoop,
    M::Scalar: Serialize + DeserializeOwned,
    M::G1Affine: Serialize + DeserializeOwned,
    M::G2Affine: Serialize + DeserializeOwned,
{
    type Param = <UnivariateKzg<M> as PolynomialCommitmentScheme<M::Scalar>>::Param;
    type ProverParam = <UnivariateKzg<M> as PolynomialCommitmentScheme<M::Scalar>>::ProverParam;
    type VerifierParam = <UnivariateKzg<M> as PolynomialCommitmentScheme<M::Scalar>>::VerifierParam;
    type Polynomial = MultilinearPolynomial<M::Scalar>;
    type Commitment = <UnivariateKzg<M> as PolynomialCommitmentScheme<M::Scalar>>::Commitment;
    type CommitmentChunk =
        <UnivariateKzg<M> as PolynomialCommitmentScheme<M::Scalar>>::CommitmentChunk;

    fn setup(poly_size: usize, batch_size: usize, rng: impl RngCore) -> Result<Self::Param, Error> {
        UnivariateKzg::<M>::setup(poly_size, batch_size, rng)
    }

    fn trim(
        param: &Self::Param,
        poly_size: usize,
        batch_size: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error> {
        UnivariateKzg::<M>::trim(param, poly_size, batch_size)
    }

    fn commit(pp: &Self::ProverParam, poly: &Self::Polynomial) -> Result<Self::Commitment, Error> {
        if pp.degree() + 1 < poly.evals().len() {
            let got = poly.evals().len() - 1;
            return Err(err_too_large_deree("commit", pp.degree(), got));
        }

        Ok(UnivariateKzg::commit_monomial(pp, poly.evals()))
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
        let num_vars = point.len();
        if pp.degree() + 1 < poly.evals().len() {
            let got = poly.evals().len() - 1;
            return Err(err_too_large_deree("open", pp.degree(), got));
        }

        if cfg!(feature = "sanity-check") {
            assert_eq!(Self::commit(pp, poly).unwrap().0, comm.0);
            assert_eq!(poly.evaluate(point), *eval);
        }

        // We set fs[0] to be the univariate polynomial with coefficients the evaluations of poly.
        // For all 0≤i<num_vars-1, fs[i+1](X) = (1-point[i])*f_e(X) + point[i]*f_o(X),
        // where fs[i](X) = f_e(X^2) + X*f_o(X^2).
        let fs = {
            let mut fs = Vec::with_capacity(num_vars);
            fs.push(UnivariatePolynomial::monomial(poly.evals().to_vec()));
            for x_i in &point[..num_vars - 1] {
                let f_i_minus_one = fs.last().unwrap().coeffs();
                let mut f_i = Vec::with_capacity(f_i_minus_one.len() >> 1);
                merge_into(&mut f_i, f_i_minus_one, x_i, 1, 0);
                fs.push(UnivariatePolynomial::monomial(f_i));
            }

            // We should now have fs[num_vars-1](point[num_vars-1]) = eval
            if cfg!(feature = "sanity-check") {
                let f_last = fs.last().unwrap();
                let x_last = point.last().unwrap();
                assert_eq!(
                    f_last[0] * (M::Scalar::ONE - x_last) + f_last[1] * x_last,
                    *eval
                );
            }

            fs
        };
        // We commit to all the univariate polys in fs.
        let comms = chain![
            [comm.clone()],
            UnivariateKzg::<M>::batch_commit_and_write(pp, &fs[1..], transcript)?
        ]
        .collect_vec();

        // We squeeze a random challenge beta and store its powers of 2 powers negated.
        let beta = transcript.squeeze_challenge();
        let points = chain![[beta], squares(beta).map(Neg::neg)]
            .take(num_vars + 1)
            .collect_vec();

        let evals = chain!([(0, 0), (0, 1)], (1..num_vars).zip(2..))
            .map(|(idx, point)| Evaluation::new(idx, point, fs[idx].evaluate(&points[point])))
            .collect_vec();
        transcript.write_field_elements(evals[1..].iter().map(Evaluation::value))?;

        // We provide a batch opening for the polys in [fs[0]. fs[0], fs[1], fs[2],...]
        // evaluated at [beta, -beta, -beta^2, -beta^4,...] respectively.
        UnivariateKzg::<M>::batch_open(pp, &fs, &comms, &points, &evals, transcript)
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
        UnivariateKzg::read_commitments(vp, num_polys, transcript)
    }

    fn verify(
        vp: &Self::VerifierParam,
        comm: &Self::Commitment,
        point: &Point<M::Scalar, Self::Polynomial>,
        eval: &M::Scalar,
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, M::Scalar>,
    ) -> Result<(), Error> {
        let num_vars = point.len();
        let comms = chain![[comm.0], transcript.read_commitments(num_vars - 1)?]
            .map(UnivariateKzgCommitment)
            .collect_vec();
        // comms is now a set of claimed commitments to the polys in fs from `open`.

        // We squeeze a random challenge beta and store its powers of 2 powers.
        let beta = transcript.squeeze_challenge();
        let squares_of_beta = squares(beta).take(num_vars).collect_vec();

        let evals = transcript.read_field_elements(num_vars)?;
        // evals are now the claimed evaluations of the [fs[0], fs[1], fs[2],...]
        // at [-beta, -beta^2, -beta^4,...] respectively from `open`.

        let one = M::Scalar::ONE;
        let two = one.double();
        // In the following `fold` the initial value is the claimed value `eval`,
        // which is fs[num_vars-1](point[num_vars-1]).
        // All subsequent values in the fold are [fs[n-1](beta^{2^(n-1)}), fs[n-2](beta^{2^(n-2)}),...,fs[0](beta)],
        // where n:=num_vars.
        let eval_0 = evals.iter().zip(&squares_of_beta).zip(point).rev().fold(
            *eval,
            |eval_pos, ((eval_neg, square_of_beta), x_i)| {
                (two * square_of_beta * eval_pos - ((one - x_i) * square_of_beta - x_i) * eval_neg)
                    * ((one - x_i) * square_of_beta + x_i).invert().unwrap()
            },
        );
        let evals = chain!([(0, 0), (0, 1)], (1..num_vars).zip(2..))
            .zip(chain![[eval_0], evals])
            .map(|((idx, point), eval)| Evaluation::new(idx, point, eval))
            .collect_vec();
        let points = chain!([beta], squares_of_beta.into_iter().map(Neg::neg)).collect_vec();

        // We provide a batch verification for the polys in [fs[0], fs[0], fs[1], fs[2],...]
        // evaluated at [beta, -beta, -beta^2, -beta^4,...] respectively.
        UnivariateKzg::<M>::batch_verify(vp, &comms, &points, &evals, transcript)
        // I think we might need more checks here! Can the prover not just commit to totally random fs[1],..,fs[num_vars-1]?
        // We seem to be making no checks about the consistency of the fs[i]'s.
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

impl<C> PolynomialCommitmentScheme<C::Scalar> for Gemini<UnivariateIpa<C>>
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
        UnivariateIpa::<C>::setup(poly_size, batch_size, rng)
    }

    fn trim(
        param: &Self::Param,
        poly_size: usize,
        batch_size: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error> {
        UnivariateIpa::<C>::trim(param, poly_size, batch_size)
    }

    fn commit(pp: &Self::ProverParam, poly: &Self::Polynomial) -> Result<Self::Commitment, Error> {
        if pp.degree() + 1 < poly.evals().len() {
            let got = poly.evals().len() - 1;
            return Err(err_too_large_deree("commit", pp.degree(), got));
        }
        let uni_poly = UnivariatePolynomial::monomial(poly.evals().to_vec());

        UnivariateIpa::commit(pp, &uni_poly)
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
        let num_vars = point.len();
        if pp.degree() + 1 < poly.evals().len() {
            let got = poly.evals().len() - 1;
            return Err(err_too_large_deree("open", pp.degree(), got));
        }

        if cfg!(feature = "sanity-check") {
            assert_eq!(Self::commit(pp, poly).unwrap().0, comm.0);
            assert_eq!(poly.evaluate(point), *eval);
        }

        // We set fs[0] to be the univariate polynomial with coefficients the evaluations of poly.
        // For all 0≤i<num_vars-1, fs[i+1](X) = (1-point[i])*f_e(X) + point[i]*f_o(X),
        // where fs[i](X) = f_e(X^2) + X*f_o(X^2).
        let fs = {
            let mut fs = Vec::with_capacity(num_vars);
            fs.push(UnivariatePolynomial::monomial(poly.evals().to_vec()));
            for x_i in &point[..num_vars - 1] {
                let f_i_minus_one = fs.last().unwrap().coeffs();
                let mut f_i = Vec::with_capacity(f_i_minus_one.len() >> 1);
                merge_into(&mut f_i, f_i_minus_one, x_i, 1, 0);
                fs.push(UnivariatePolynomial::monomial(f_i));
            }

            // We should now have fs[num_vars-1](point[num_vars-1]) = eval
            if cfg!(feature = "sanity-check") {
                let f_last = fs.last().unwrap();
                let x_last = point.last().unwrap();
                assert_eq!(
                    f_last[0] * (C::Scalar::ONE - x_last) + f_last[1] * x_last,
                    *eval
                );
            }

            fs
        };
        // We commit to all the univariate polys in fs.
        let comms = chain![
            [comm.clone()],
            UnivariateIpa::<C>::batch_commit_and_write(pp, &fs[1..], transcript)?
        ]
        .collect_vec();

        // We squeeze a random challenge beta and store its powers of 2 powers negated.
        let beta = transcript.squeeze_challenge();
        let points = chain![[beta], squares(beta).map(Neg::neg)]
            .take(num_vars + 1)
            .collect_vec();

        let evals = chain!([(0, 0), (0, 1)], (1..num_vars).zip(2..))
            .map(|(idx, point)| Evaluation::new(idx, point, fs[idx].evaluate(&points[point])))
            .collect_vec();
        transcript.write_field_elements(evals[1..].iter().map(Evaluation::value))?;

        // We provide a batch opening for the polys in [fs[0]. fs[0], fs[1], fs[2],...]
        // evaluated at [beta, -beta, -beta^2, -beta^4,...] respectively.
        UnivariateIpa::<C>::batch_open(pp, &fs, &comms, &points, &evals, transcript)
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
        let comms = chain![[comm.0], transcript.read_commitments(num_vars - 1)?]
            .map(UnivariateIpaCommitment)
            .collect_vec();
        // comms is now a set of claimed commitments to the polys in fs from `open`.

        // We squeeze a random challenge beta and store its powers of 2 powers.
        let beta = transcript.squeeze_challenge();
        let squares_of_beta = squares(beta).take(num_vars).collect_vec();

        let evals = transcript.read_field_elements(num_vars)?;
        // evals are now the claimed evaluations of the [fs[0], fs[1], fs[2],...]
        // at [-beta, -beta^2, -beta^4,...] respectively from `open`.

        let one = C::Scalar::ONE;
        let two = one.double();
        // In the following `fold` the initial value is the claimed value `eval`,
        // which is fs[num_vars-1](point[num_vars-1]).
        // All subsequent values in the fold are [fs[n-1](beta^{2^(n-1)}), fs[n-2](beta^{2^(n-2)}),...,fs[0](beta)],
        // where n:=num_vars.
        let eval_0 = evals.iter().zip(&squares_of_beta).zip(point).rev().fold(
            *eval,
            |eval_pos, ((eval_neg, square_of_beta), x_i)| {
                (two * square_of_beta * eval_pos - ((one - x_i) * square_of_beta - x_i) * eval_neg)
                    * ((one - x_i) * square_of_beta + x_i).invert().unwrap()
            },
        );
        let evals = chain!([(0, 0), (0, 1)], (1..num_vars).zip(2..))
            .zip(chain![[eval_0], evals])
            .map(|((idx, point), eval)| Evaluation::new(idx, point, eval))
            .collect_vec();
        let points = chain!([beta], squares_of_beta.into_iter().map(Neg::neg)).collect_vec();

        // We provide a batch verification for the polys in [fs[0], fs[0], fs[1], fs[2],...]
        // evaluated at [beta, -beta, -beta^2, -beta^4,...] respectively.
        UnivariateIpa::<C>::batch_verify(vp, &comms, &points, &evals, transcript)
        // I think we might need more checks here! Can the prover not just commit to totally random fs[1],..,fs[num_vars-1]?
        // We seem to be making no checks about the consistency of the fs[i]'s.
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

#[cfg(test)]
mod test {
    use crate::{
        pcs::{
            multilinear::gemini::Gemini,
            test::{run_batch_commit_open_verify, run_commit_open_verify},
            univariate::{UnivariateIpa, UnivariateKzg},
        },
        util::transcript::Keccak256Transcript,
    };
    use halo2_curves::{bn256::Bn256, grumpkin};

    type GemKzgPcs = Gemini<UnivariateKzg<Bn256>>;
    type GemIpaPcs = Gemini<UnivariateIpa<grumpkin::G1Affine>>;

    #[test]
    fn commit_open_verify() {
        run_commit_open_verify::<_, GemKzgPcs, Keccak256Transcript<_>>();
        run_commit_open_verify::<_, GemIpaPcs, Keccak256Transcript<_>>();
    }

    #[test]
    fn batch_commit_open_verify() {
        run_batch_commit_open_verify::<_, GemKzgPcs, Keccak256Transcript<_>>();
        run_batch_commit_open_verify::<_, GemIpaPcs, Keccak256Transcript<_>>();
    }
}
