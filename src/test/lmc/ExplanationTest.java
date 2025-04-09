/*
 * Copyright (C)  2022 Franck van Breugel
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

package test.lmc;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.junit.jupiter.api.Test;

import explainability.lmc.Explanation;
import explainability.lmc.LabelledMarkovChain;
import static explainability.lmc.Constants.ACCURACY;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.RepeatedTest;

/**
 * Tests the Explanation class.
 *
 * @author Anto Nanah Ji
 * @author Franck van Breugel
 */
public class ExplanationTest {
	private static final int NUMBER = 1000;

	/**
	 * Tests policy to dot.
	 */
//	@RepeatedTest(NUMBER)
	@Test
	void testPolicyToDot() {
		final int NUMBER_OF_STATES = 10;
		final int NUMBER_OF_LABELS = 2;
		LabelledMarkovChain chain = LabelledMarkovChain.randomUniform(NUMBER_OF_STATES, NUMBER_OF_LABELS);
		Explanation explanation = new Explanation(chain);
		String representation = explanation.policyToDot(1, 0);
		System.out.println(representation);
	}

	/**
	 * Tests the computation of λ1(s, t) with two different methods: Power and DFS.
	 */
//	@RepeatedTest(NUMBER)
	@Test
	void testLambdaOneDFSAndPower() {
		final int NUMBER_OF_STATES = 10;
		final int NUMBER_OF_LABELS = 2;

		LabelledMarkovChain chain = LabelledMarkovChain.randomUniform(NUMBER_OF_STATES, NUMBER_OF_LABELS);
		Explanation explanation = new Explanation(chain);
		explanation.InitializeOneMatrixVectors();
		explanation.lambdaOnePower();
		double[] expectedLengthsPower = explanation.getOneExpectedLengths();

		double[][] policy = explanation.getPolicy();
		boolean[] bisimilar = explanation.getBisimilar();
		double[] distance = explanation.getDistance();
		boolean[] differentLabels = explanation.getDifferentLabels();
		double[] expectedLengthsDFS = this.lambdaOneDFS(NUMBER_OF_STATES, bisimilar, differentLabels, policy, distance);

		for (int j = 0; j < expectedLengthsPower.length; j++) {
			double diff = Math.abs(expectedLengthsPower[j] - expectedLengthsDFS[j]);
			assertTrue(diff < ACCURACY, "The result from the power method is equal to the DFS.");
		}

	}

	/**
	 * Tests the computation of λ1(s, t) with two different methods: Power and
	 * System of equations.
	 */
//	@RepeatedTest(NUMBER)
	@Test
	void testLambdaOnePowerAndSystemOfEquations() {
		final int NUMBER_OF_STATES = 10;
		final int NUMBER_OF_LABELS = 2;

		LabelledMarkovChain chain = LabelledMarkovChain.randomUniform(NUMBER_OF_STATES, NUMBER_OF_LABELS);
		Explanation explanation = new Explanation(chain);

		double[][] policy = explanation.getPolicy();
		boolean[] bisimilar = explanation.getBisimilar();
		double[] distance = explanation.getDistance();
		boolean[] differentLabels = explanation.getDifferentLabels();
		double[] expectedLengthsSystemOfEquations = this.lambdaOneSystemOfEquations(NUMBER_OF_STATES, bisimilar,
				differentLabels, policy, distance);

		explanation.InitializeOneMatrixVectors();
		explanation.lambdaOnePower();
		double[] expectedLengthsPower = explanation.getOneExpectedLengths();

		for (int j = 0; j < expectedLengthsSystemOfEquations.length; j++) {
			double diff = Math.abs(expectedLengthsSystemOfEquations[j] - expectedLengthsPower[j]);
			assertTrue(diff < ACCURACY, "The result from the power method is equal to the SystemOfEquations.");
		}

	}

	/**
	 * Tests the computation of λ0(s, t) with two different methods: Power and
	 * System of equations.
	 */
//	@RepeatedTest(NUMBER)
	@Test
	void testLambdaZeroPowerAndSystemOfEquations() {
		final int NUMBER_OF_STATES = 10;
		final int NUMBER_OF_LABELS = 2;

		LabelledMarkovChain chain = LabelledMarkovChain.randomUniform(NUMBER_OF_STATES, NUMBER_OF_LABELS);
		Explanation explanation = new Explanation(chain);

		double[][] policy = explanation.getPolicy();
		boolean[] bisimilar = explanation.getBisimilar();
		double[] distance = explanation.getDistance();
		boolean[] differentLabels = explanation.getDifferentLabels();
		double[] expectedLengthsSystemOfEquations = this.lambdaZeroSystemOfEquations(NUMBER_OF_STATES, bisimilar,
				differentLabels, policy, distance);

		explanation.InitializeZeroMatrixVectors();
		explanation.lambdaZeroPower();
		double[] expectedLengthsPower = explanation.getZeroExpectedLengths();

		for (int j = 0; j < expectedLengthsSystemOfEquations.length; j++) {
			double diff = Math.abs(expectedLengthsSystemOfEquations[j] - expectedLengthsPower[j]);
			assertTrue(diff < ACCURACY, "The result from the power method is equal to the SystemOfEquations.");
		}
	}


	/**
	 * Tests the computation of one maximal policy by ensuring that, for all state
	 * pairs in the resulting policy, the expected length is greater than or equal
	 * to its initial length.
	 */
//	@RepeatedTest(NUMBER)
	@Test
	void testOneMaximalB() {
		final int NUMBER_OF_STATES = 7;
		final int NUMBER_OF_LABELS = 2;

		LabelledMarkovChain chain = LabelledMarkovChain.randomUniform(NUMBER_OF_STATES, NUMBER_OF_LABELS);
		Explanation explanation = new Explanation(chain);
		explanation.InitializeOneMatrixVectors();
		explanation.lambdaOnePower();
		double[] initial = explanation.getOneExpectedLengths();

		explanation.oneMaximize();
		double[] maximal = explanation.getOneExpectedLengths();

		for (int j = 0; j < initial.length; j++) {
			assertTrue(maximal[j] >= initial[j], maximal[j] + " >= " + initial[j]);
		}

		System.out.println("Initial ONE expected lengths are less than or equal to the final ONE expected lengths!\n");
	}

	/**
	 * Tests the computation of one maximal policy by ensuring that, for all state
	 * pairs in the resulting policy, λ1(s, t) = P(s, t) · (1 + λ1).
	 */
//	@RepeatedTest(NUMBER)
	@Test
	void testOneMaximalC() {
		final int NUMBER_OF_STATES = 7;
		final int NUMBER_OF_LABELS = 2;

		LabelledMarkovChain chain = LabelledMarkovChain.randomUniform(NUMBER_OF_STATES, NUMBER_OF_LABELS);
		Explanation explanation = new Explanation(chain);
		explanation.oneMaximize();

		double[][] policy = explanation.getPolicy();
		boolean[] bisimilar = explanation.getBisimilar();
		boolean[] differentLabels = explanation.getDifferentLabels();
		double[] distance = explanation.getDistance();
		double[] expectedLengths = explanation.getOneExpectedLengths();
		double sum = 0;

		for (int s = 0; s < NUMBER_OF_STATES; s++) {
			for (int t = 0; t < s; t++) {
				if (bisimilar[s * NUMBER_OF_STATES + t]) {
					// do nothing
				} else if (differentLabels[s * NUMBER_OF_STATES + t]) {
					// do nothing
				} else {
					sum = 0;
					for (int u = 0; u < NUMBER_OF_STATES; u++) {
						for (int v = 0; v < NUMBER_OF_STATES; v++) {
							if (bisimilar[u * NUMBER_OF_STATES + v]) {
								// do nothing
							} else if (differentLabels[u * NUMBER_OF_STATES + v]) {
								sum += policy[s * NUMBER_OF_STATES + t][u * NUMBER_OF_STATES + v];
							} else {
								sum += policy[s * NUMBER_OF_STATES + t][u * NUMBER_OF_STATES + v]
										* (distance[u * NUMBER_OF_STATES + v]
												+ expectedLengths[u * NUMBER_OF_STATES + v]);
							}

						}
					}
					double diff = Math.abs(expectedLengths[s * NUMBER_OF_STATES + t] - sum);
					assertTrue(diff < ACCURACY,
							"λ1(" + s + "," + t + ") = ∑ P(" + s + "," + t + ")(u, v) (d(u, v) + λ1) \n");
				}
			}
		}
		System.out.println("λ1(s, t) = ∑ P(s, t)(u, v) (d(u, v) + λ1) for all (s, t)!\n");
	}

	/**
	 * Tests the computation of zero minimal policy by ensuring that, for all state
	 * pairs in the resulting policy, the expected length is less than or equal to
	 * its initial length.
	 */
//	@RepeatedTest(NUMBER)
	@Test
	void testZeroMinimalB() {
		final int NUMBER_OF_STATES = 7;
		final int NUMBER_OF_LABELS = 2;

		LabelledMarkovChain chain = LabelledMarkovChain.randomUniform(NUMBER_OF_STATES, NUMBER_OF_LABELS);

		Explanation explanation = new Explanation(chain);
		explanation.InitializeZeroMatrixVectors();
		explanation.lambdaZeroPower();
		double[] initial = explanation.getZeroExpectedLengths();

		explanation.zeroMinimize();
		double[] minimal = explanation.getZeroExpectedLengths();

		for (int j = 0; j < initial.length; j++) {
			assertTrue(minimal[j] <= initial[j], minimal[j] + " <= " + initial[j]);

		}

		System.out.println("Initial ZERO expected lengths are less than or equal to the final ZERO expected lengths!\n");
	}

	/**
	 * Tests the computation of zero minimal policy by ensuring that, for all state
	 * pairs in the resulting policy, λ0(s, t) = P(s, t) · (1 + λ0).
	 */
//	@RepeatedTest(NUMBER)
	@Test
	void testZeroMinimalC() {
		final int NUMBER_OF_STATES = 7;
		final int NUMBER_OF_LABELS = 2;

		LabelledMarkovChain chain = LabelledMarkovChain.randomUniform(NUMBER_OF_STATES, NUMBER_OF_LABELS);
		Explanation explanation = new Explanation(chain);
		explanation.zeroMinimize();

		double[][] policy = explanation.getPolicy();
		boolean[] bisimilar = explanation.getBisimilar();
		double[] distance = explanation.getDistance();
		boolean[] differentLabels = explanation.getDifferentLabels();
		double[] expectedLengths = explanation.getZeroExpectedLengths();
		double sum = 0;

		for (int s = 0; s < NUMBER_OF_STATES; s++) {
			for (int t = 0; t < s; t++) {
				if (bisimilar[s * NUMBER_OF_STATES + t]) {
					// do nothing
				} else if (differentLabels[s * NUMBER_OF_STATES + t]) {
					// do nothing
				} else {
					sum = 0;
					for (int u = 0; u < NUMBER_OF_STATES; u++) {
						for (int v = 0; v < NUMBER_OF_STATES; v++) {
							if (bisimilar[u * NUMBER_OF_STATES + v]) {
								sum += policy[s * NUMBER_OF_STATES + t][u * NUMBER_OF_STATES + v];
							} else if (differentLabels[u * NUMBER_OF_STATES + v]) {
								// do nothing
							} else {
								sum += policy[s * NUMBER_OF_STATES + t][u * NUMBER_OF_STATES + v]
										* (1 - distance[u * NUMBER_OF_STATES + v]
												+ expectedLengths[u * NUMBER_OF_STATES + v]);
							}

						}
					}
					double diff = Math.abs(expectedLengths[s * NUMBER_OF_STATES + t] - sum);
					assertTrue(diff < ACCURACY,
							"λ0(" + s + "," + t + ") =  ∑ P(" + s + "," + t + ")(u, v) ((1 - d(u, v)) + λ0) \n");
				}
			}
		}
		System.out.println("λ0(s, t) = ∑ P(s, t)(u, v) ((1 - d(u, v) + λ0) for all (s, t)!\n");
	}

	/**
	 * Computes λ1 for all the states in S1? using the DFS method. The computation
	 * result is stored in oneExpectedLengths array.
	 *
	 * @param numberOfStates  The number of states of the labelled Markov chain.
	 * @param bisimilar       For states s, t, bisimilar[s * numberOfStates + t] ==
	 *                        s and t are probabilistic bisimilar
	 * @param differentLabels For states s, t, differentLabels[s * numberOfStates +
	 *                        t] == s and t have different labels
	 * @param policy          An optimal policy
	 * @return λ1(s, t) for all (s, t) in S1?
	 */
	private double[] lambdaOneDFS(int numberOfStates, boolean[] bisimilar, boolean[] differentLabels, double[][] policy,
			double[] distances) {
		double[] lengths = new double[numberOfStates * numberOfStates];
		for (int s = 0; s < numberOfStates; s++) {
			for (int t = 0; t < s; t++) {
				if (bisimilar[s * numberOfStates + t]) {
					// Do nothing
				} else if (differentLabels[s * numberOfStates + t]) {
					// Do nothing
				} else {
					if (lengths[s * numberOfStates + t] == 0) {
						List<List<Integer>> ls = new ArrayList<>();
						List<List<Double>> lsd = new ArrayList<>();
						List<Integer> set = new ArrayList<>();
						set.add(s * numberOfStates + t);

						lambdaOneDFSHelper(s, t, numberOfStates, ls, lsd, set, bisimilar, differentLabels, policy,
								distances);

						int size = ls.size();

						double[] constants = new double[size];

						for (int i = 0; i < size; i++) {
							constants[set.indexOf(ls.get(i).get(0))] = lsd.get(i).get(0);
						}

						double[][] coefficients = new double[size][size];

						for (int i = 0; i < size; i++) {
							int index = set.indexOf(ls.get(i).get(0));
							coefficients[index][index] = 1;
							int st = ls.get(i).get(0);

							for (int k = 1; k < ls.get(i).size(); k++) {
								if (ls.get(i).get(k) == st) {
									coefficients[index][index] = 1.0 - lsd.get(i).get(k);
								} else {
									coefficients[index][set.indexOf(ls.get(i).get(k))] = -1.0 * lsd.get(i).get(k);
								}
							}
						}

						RealVector solution = this.LUDecompositionSolver(coefficients, constants);

						for (int ss = 0; ss < numberOfStates; ss++) {
							for (int tt = 0; tt < ss; tt++) {
								int index = set.indexOf(ss * numberOfStates + tt);
								if (index > -1) {
									double value = solution.getEntry(index);
									lengths[ss * numberOfStates + tt] = value;
									lengths[tt * numberOfStates + ss] = value;
								}
							}
						}
					}
				}
			}
		}

		return lengths;
	}

	/**
	 * Recursively constructs the λ1 equations for all states in S1? that can be
	 * reached from s and t.
	 *
	 * @param s   a state
	 * @param t   a state
	 * @param ls  a list of lists to store the states in S? that are reachable from
	 *            s and t
	 * @param lsd a list of lists to store the probabilities
	 * @param set a list to keep track of visited states @pre. s < t
	 */
	private void lambdaOneDFSHelper(int s, int t, int numberOfStates, List<List<Integer>> ls, List<List<Double>> lsd,
			List<Integer> set, boolean[] bisimilar, boolean[] differentLabels, double[][] policy, double[] distances) {
		double[] coupling = policy[s * numberOfStates + t];
		List<Integer> l = new ArrayList<>();
		List<Double> ld = new ArrayList<>();
		l.add(s * numberOfStates + t);
		ld.add(0.0);
		for (int u = 0; u < numberOfStates; u++) {
			for (int v = 0; v < numberOfStates; v++) {

				double value = coupling[u * numberOfStates + v];
				int x = u;
				int y = v;

				if (u < v) {
					x = v;
					y = u;
				}

				if (set.contains(x * numberOfStates + y)) {
					if (value > 0) {
						// add to the formula is missing
						ld.set(0, (ld.get(0) + (value * distances[u * numberOfStates + v])));
						ld.add(value);
						l.add(x * numberOfStates + y);
					}
					continue;
				}

				if (value > 0) {
					if (bisimilar[u * numberOfStates + v]) {
						// do nothing
					} else if (differentLabels[u * numberOfStates + v]) {
						ld.set(0, (ld.get(0) + (value * distances[u * numberOfStates + v])));
					} else {
						set.add(x * numberOfStates + y);
						ld.set(0, (ld.get(0) + (value * distances[u * numberOfStates + v])));
						if (s == x && t == y) {
							ld.add(1.0 - value);
							l.add(x * numberOfStates + y);
						} else {
							ld.add(value);
							l.add(x * numberOfStates + y);
							lambdaOneDFSHelper(x, y, numberOfStates, ls, lsd, set, bisimilar, differentLabels, policy,
									distances);
						}
					}
				}
			}
		}
		ls.add(l);
		lsd.add(ld);
	}

	/**
	 * Computes λ1 for all the states in S1? using the system of equations method.
	 * The computation result is stored in oneExpectedLengths array.
	 *
	 * @param numberOfStates  The number of states of the labelled Markov chain.
	 * @param bisimilar       For states s, t, bisimilar[s * numberOfStates + t] ==
	 *                        s and t are probabilistic bisimilar
	 * @param differentLabels For states s, t, differentLabels[s * numberOfStates +
	 *                        t] == s and t have different labels
	 * @param policy          An optimal policy
	 * @return λ1(s, t) for all (s, t) in S1?
	 */
	private double[] lambdaOneSystemOfEquations(int numberOfStates, boolean[] bisimilar, boolean[] differentLabels,
			double[][] policy, double[] distances) {
		int size = numberOfStates * numberOfStates;
		double[] constants = new double[size];
		double[][] coefficients = new double[size][size];

		for (int s = 0; s < numberOfStates; s++) {
			for (int t = 0; t < s; t++) {
				if (bisimilar[s * numberOfStates + t]) {

				} else if (differentLabels[s * numberOfStates + t]) {

				} else {
					double sum = 0;

					for (int u = 0; u < numberOfStates; u++) {
						for (int v = 0; v < numberOfStates; v++) {
							if (!bisimilar[u * numberOfStates + v]) {
								sum += (distances[u * numberOfStates + v]
										* policy[s * numberOfStates + t][u * numberOfStates + v]);
							}
						}
					}
					constants[s * numberOfStates + t] = sum;
					constants[t * numberOfStates + s] = sum;
				}
			}
		}

		for (int s = 0; s < numberOfStates; s++) {
			for (int t = 0; t <= s; t++) {
				coefficients[s * numberOfStates + t][s * numberOfStates + t] = 1;
				coefficients[t * numberOfStates + s][t * numberOfStates + s] = 1;

				if (bisimilar[s * numberOfStates + t]) {

				} else if (differentLabels[s * numberOfStates + t]) {

				} else {
					for (int u = 0; u < numberOfStates; u++) {
						for (int v = 0; v < numberOfStates; v++) {
							double value = policy[s * numberOfStates + t][u * numberOfStates + v];

							if (s == u && t == v) {
								coefficients[s * numberOfStates + t][u * numberOfStates + v] = 1 - value;
								coefficients[t * numberOfStates + s][v * numberOfStates + u] = 1 - value;

							} else {
								if (value > 0 && !differentLabels[u * numberOfStates + v]) {
									coefficients[s * numberOfStates + t][u * numberOfStates + v] = -1 * value;
									coefficients[t * numberOfStates + s][v * numberOfStates + u] = -1 * value;
								}
							}
						}
					}
				}
			}
		}

		return this.LUDecompositionSolver(coefficients, constants).toArray();
	}

	/**
	 * Computes LUDecomposition.
	 *
	 * @param coefficients matrix
	 * @param constants    vector
	 * @return a solution for LUDecomposition
	 */
	private RealVector LUDecompositionSolver(double[][] coefficients, double[] constants) {
		RealMatrix matrix = MatrixUtils.createRealMatrix(coefficients);
		RealVector vector = new ArrayRealVector(constants);

		DecompositionSolver solver = new LUDecomposition(matrix).getSolver();
		RealVector solution = solver.solve(vector);

		return solution;
	}

	/**
	 * Computes λ0 for all the states in S0? using the system of equations method.
	 * The computation result is stored in zeroExpectedLengths array.
	 *
	 * @param numberOfStates  The number of states of the labelled Markov chain.
	 * @param bisimilar       For states s, t, bisimilar[s * numberOfStates + t] ==
	 *                        s and t are probabilistic bisimilar
	 * @param differentLabels For states s, t, differentLabels[s * numberOfStates +
	 *                        t] == s and t have different labels
	 * @param policy          An optimal policy
	 * @return λ0(s, t) for all (s, t) in S0?
	 */
	private double[] lambdaZeroSystemOfEquations(int numberOfStates, boolean[] bisimilar, boolean[] differentLabels,
			double[][] policy, double[] distances) {
		int size = numberOfStates * numberOfStates;
		double[] constants = new double[size];
		double[][] coefficients = new double[size][size];

		for (int s = 0; s < numberOfStates; s++) {
			for (int t = 0; t < s; t++) {
				if (bisimilar[s * numberOfStates + t]) {

				} else if (differentLabels[s * numberOfStates + t]) {

				} else {
					double sum = 0;

					for (int u = 0; u < numberOfStates; u++) {
						for (int v = 0; v < numberOfStates; v++) {
							if (!differentLabels[u * numberOfStates + v]) {
								sum += ((1 - distances[u * numberOfStates + v])
										* policy[s * numberOfStates + t][u * numberOfStates + v]);
							}
						}
					}
					constants[s * numberOfStates + t] = sum;
					constants[t * numberOfStates + s] = sum;
				}
			}
		}

		for (int s = 0; s < numberOfStates; s++) {
			for (int t = 0; t <= s; t++) {
				coefficients[s * numberOfStates + t][s * numberOfStates + t] = 1;
				coefficients[t * numberOfStates + s][t * numberOfStates + s] = 1;

				if (bisimilar[s * numberOfStates + t]) {

				} else if (differentLabels[s * numberOfStates + t]) {

				} else {
					for (int u = 0; u < numberOfStates; u++) {
						for (int v = 0; v < numberOfStates; v++) {
							double value = policy[s * numberOfStates + t][u * numberOfStates + v];

							if (s == u && t == v) {
								coefficients[s * numberOfStates + t][u * numberOfStates + v] = 1 - value;
								coefficients[t * numberOfStates + s][v * numberOfStates + u] = 1 - value;

							} else {
								if (value > 0 && !bisimilar[u * numberOfStates + v]) {
									coefficients[s * numberOfStates + t][u * numberOfStates + v] = -1 * value;
									coefficients[t * numberOfStates + s][v * numberOfStates + u] = -1 * value;
								}
							}
						}
					}
				}
			}
		}

		return this.LUDecompositionSolver(coefficients, constants).toArray();
	}

}
