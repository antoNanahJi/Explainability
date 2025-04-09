/*
 * Copyright (C)  2024  Emily Vlasman, Anto Nanah Ji, and Franck van Breugel
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

package explainability.lmc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.linear.LinearConstraint;
import org.apache.commons.math3.optim.linear.LinearConstraintSet;
import org.apache.commons.math3.optim.linear.LinearObjectiveFunction;
import org.apache.commons.math3.optim.linear.NonNegativeConstraint;
import org.apache.commons.math3.optim.linear.Relationship;
import org.apache.commons.math3.optim.linear.SimplexSolver;
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;
import org.apache.commons.math3.optimization.linear.*;
import org.apache.commons.math3.stat.regression.ModelSpecificationException;

import static explainability.lmc.Constants.ACCURACY;

import explainability.lmc.compute.SimplePolicyIteration;

/**
 * Generates an optimal policy that explains the probabilistic bisimilarity
 * distances of a labelled Markov chain.
 *
 * @author Emily Vlasman
 * @author Anto Nanah Ji
 * @author Franck van Breugel
 */
public class Explanation {
	/**
	 * Labelled Markov chain
	 */
	private LabelledMarkovChain chain;

	/**
	 * The number of states of the labelled Markov chain.
	 */
	private final int numberOfStates;

	/**
	 * For states s, t, bisimilar[s * numberOfStates + t] == s and t are
	 * probabilistic bisimilar.
	 */
	private boolean[] bisimilar;

	/**
	 * For states s, t, differentLabels[s * numberOfStates + t] == s and t have
	 * different labels.
	 */
	private boolean[] differentLabels;

	/**
	 * The distances of the labelled Markov chain: distance[s * numberOfStates + t]
	 * = distance of s and t.
	 */
	private double[] distance;

	/**
	 * An optimal policy.
	 */
	private double[][] policy;

	/**
	 * Computes the probabilistic bisimilarity distances by means of simple policy
	 * iteration.
	 */
	private SimplePolicyIteration iteration;

	/**
	 * The 1-Maximal expected lengths of the labelled Markov chain:
	 * oneExpectedLengths[s * numberOfStates + t] = λ1(s, t).
	 */
	private double[] oneExpectedLengths;

	/**
	 * The O-Minimal expected lengths of the labelled Markov chain:
	 * zeroExpectedLengths[s * numberOfStates + t] = λ0(s, t).
	 */
	private double[] zeroExpectedLengths;

	/**
	 * Matrix A for the power method
	 */
	private RealMatrix matrixA;

	/**
	 * Vector B for the power method
	 */
	private RealVector vectorB;
	
	/**
	 * Vector D for the power method
	 */
	private RealVector vectorD;

	/**
	 * Initializes this explanation for the given labelled Markov chain.
	 *
	 * @param chain a labelled Markov chain.
	 */
	public Explanation(LabelledMarkovChain chain) {
		this.numberOfStates = chain.getNumberOfStates();
	    iteration = new SimplePolicyIteration(chain);
		this.bisimilar = new boolean[numberOfStates * numberOfStates];
		this.differentLabels = new boolean[numberOfStates * numberOfStates];
		this.policy = new double[numberOfStates * numberOfStates][numberOfStates * numberOfStates];
		this.distance = new double[numberOfStates * numberOfStates];
		this.oneExpectedLengths = new double[numberOfStates * numberOfStates];
		this.zeroExpectedLengths = new double[numberOfStates * numberOfStates];
		this.chain = chain;
	}
	
	/**
	 * getOneExpectedLengths()[s * chain.getNumberOfStates() + t] == λ1(s, t).
	 * 
	 * @return λ1(s, t) for all states (s, t).
	 */
	public double[] getOneExpectedLengths() {
		return this.oneExpectedLengths;
	}

	/**
	 * getZeroExpectedLengths()[s * chain.getNumberOfStates() + t] == λ0(s, t).
	 * 
	 * @return λ0(s, t) for all states (s, t).
	 */
	public double[] getZeroExpectedLengths() {
		return this.zeroExpectedLengths;
	}

	/**
	 * Returns whether states are probabilistic bisimilar. For states s, t,
	 * getBisimilar()[s * chain.getNumberOfStates() + t] == s and t are
	 * probabilistic bisimilar.
	 *
	 * @return Returns whether states are probabilistic bisimilar.
	 */
	public boolean[] getBisimilar() {
		return this.bisimilar;
	}

	/**
	 * Returns whether states have different labels. For states s, t,
	 * getDifferentLabels()[s * chain.getNumberOfStates() + t] == s and t have
	 * different labels.
	 *
	 * @return Returns whether states have different labels.
	 */
	public boolean[] getDifferentLabels() {
		return this.differentLabels;
	}

	/**
	 * Returns the computed policy.
	 *
	 * @return an optimal policy.
	 */
	public double[][] getPolicy() {
		return this.policy;
	}

	/**
	 * Returns the probabilistic bisimilarity distances.
	 *
	 * @return the probabilistic bisimilarity distances.
	 */
	public double[] getDistance() {
		return this.distance;
	}
	
	/**
	 * Sets the policy to an optimal one by means of policy iteration algorithm. 
	 */	
	public void optimalPolicy() {
		iteration.compute();
		this.bisimilar = iteration.getBisimilar();
		this.differentLabels = iteration.getDifferentLabels();
		this.policy = iteration.getPolicy(); // optimal policy
		this.distance = iteration.getDistance();
	}

	/**
	 * Sets the policy to an optimal one that maximizes the expected length to pairs
	 * of states with different labels.
	 */
	public void oneMaximize() {
		// Initialize the required matrix A and vector B for power method
		this.InitializeOneMatrixVectors();
		// Compute lambda1
		this.lambdaOnePower();		

		boolean maximal;
		do {
			maximal = true;
			for (int s = 0; s < this.numberOfStates; s++) {
				for (int t = 0; t < this.numberOfStates; t++) {
					if (this.bisimilar[s * this.numberOfStates + t]) {
						// Do nothing
					} else if (this.differentLabels[s * this.numberOfStates + t]) {
						// Do nothing
					} else {
						PointValuePair solution = this.getOneOptimalSolution(s, t);
						double oldLambdaP = this.oneExpectedLengths[s * this.numberOfStates + t];

						double diff = solution.getValue() - oldLambdaP;
						if (diff > ACCURACY) {
							
							maximal = false;
							// Update the policy
							this.policy[s * this.numberOfStates + t] = solution.getPoint();

							// Update the matrix A and vector B
							double a = 0;
							for (int u = 0; u < this.numberOfStates; u++) {
								for (int v = 0; v < this.numberOfStates; v++) {
									if (this.differentLabels[u * this.numberOfStates + v]) {
										a += this.policy[s * this.numberOfStates + t][u * this.numberOfStates + v];

									}
									if (!this.bisimilar[u * this.numberOfStates + v]
											&& !this.differentLabels[u * this.numberOfStates + v]) {

										this.matrixA.setEntry(s * this.numberOfStates + t, u * this.numberOfStates + v,
												solution.getPoint()[u * this.numberOfStates + v]);

									}
								}
							}

							a += this.matrixA.getRowVector(s * this.numberOfStates + t).dotProduct(this.vectorD);

							this.vectorB.setEntry(s * this.numberOfStates + t, a);

							// Compute lambda1
							this.lambdaOnePower();

						}
					}
				}
			}
		} while (!maximal);
	}

	/**
	 * Sets the policy to an optimal one that maximizes the expected length to pairs
	 * of states with different labels and minimizes the expected length to pairs of
	 * states that are probabilistic bisimilar.
	 */
	public void zeroMinimize() {
		// Initializes the required matrix A and vector B for power method
		this.InitializeZeroMatrixVectors();
		// Compute lambda0
		this.lambdaZeroPower();

		boolean minimal;
		do {
			minimal = true;
			for (int s = 0; s < this.numberOfStates; s++) {
				for (int t = 0; t < this.numberOfStates; t++) {
					if (this.bisimilar[s * this.numberOfStates + t]) {
						// Do nothing
					} else if (this.differentLabels[s * this.numberOfStates + t]) {
						// Do nothing
					} else {

						PointValuePair solution = this.getZeroOptimalSolution(s, t);
						double oldLambdaP = this.zeroExpectedLengths[s * this.numberOfStates + t];

						double diff = oldLambdaP - solution.getValue();
						if (diff > ACCURACY) {
							minimal = false;
							// Update the policy
							this.policy[s * this.numberOfStates + t] = solution.getPoint();

							// Update the matrix A and vector B
							double a = 0;
							for (int u = 0; u < this.numberOfStates; u++) {
								for (int v = 0; v < this.numberOfStates; v++) {
									if (this.bisimilar[u * this.numberOfStates + v]) {
										a += this.policy[s * this.numberOfStates + t][u * this.numberOfStates + v];

									}
									if (!this.bisimilar[u * this.numberOfStates + v]
											&& !this.differentLabels[u * this.numberOfStates + v]) {

										this.matrixA.setEntry(s * this.numberOfStates + t, u * this.numberOfStates + v,
												solution.getPoint()[u * this.numberOfStates + v]);

									}
								}
							}

							a += this.matrixA.getRowVector(s * this.numberOfStates + t).dotProduct(this.vectorD);

							this.vectorB.setEntry(s * this.numberOfStates + t, a);

							// Compute lambda0
							this.lambdaZeroPower();
						}
					}

				}
			}

		} while (!minimal);
	}

	/**
	 * Returns an optimal solution for the given states. For states s and t,
	 * getOneOptimalSolution(s, t) == ΛP1(λ1)(s, t).
	 * 
	 * @param s a state
	 * @param t a state
	 * @return ΛP1(λ1)(s, t)
	 */
	public PointValuePair getOneOptimalSolution(int s, int t) {
		// Objective function
		double[] coefficient = new double[this.numberOfStates * this.numberOfStates];
		for (int v = 0; v < this.numberOfStates; v++) {
			for (int u = 0; u < this.numberOfStates; u++) {
				if (this.bisimilar[u * this.numberOfStates + v]) {
					// Do nothing
				} else if (this.differentLabels[u * this.numberOfStates + v]) {
					coefficient[u * numberOfStates + v] = 1;
				} else {
					coefficient[u * numberOfStates + v] = (this.distance[u * numberOfStates + v]
							+ this.oneExpectedLengths[u * numberOfStates + v]);
				}
			}
		}

		LinearObjectiveFunction objective = new LinearObjectiveFunction(coefficient, 0);
		
		// Constraints
		Collection<LinearConstraint> constraints = this.getCommonConstraints(s, t);

		// Simplex solver
		SimplexSolver solver = new SimplexSolver();		
		return solver.optimize(objective, new LinearConstraintSet(constraints), GoalType.MAXIMIZE,
	                		new NonNegativeConstraint(true));
	}

	/**
	 * Generates and returns the common constraints for ΛP1(λ1)(s, t) and ΛP0(λ0)(s,
	 * t).
	 * 
	 * @param s a state
	 * @param t a state
	 * @return a collection of linear constraints
	 */
	private Collection<LinearConstraint> getCommonConstraints(int s, int t) {
		// Common constraints
		double[] coefficient = new double[this.numberOfStates * this.numberOfStates];
		Collection<LinearConstraint> constraints = new ArrayList<LinearConstraint>();
		for (int u = 0; u < this.numberOfStates; u++) {
			Arrays.fill(coefficient, 0);
			for (int v = 0; v < this.numberOfStates; v++) {
				coefficient[u * this.numberOfStates + v] = 1;
			}
			constraints.add(new LinearConstraint(coefficient, Relationship.EQ, this.chain.getProbability(s, u)));
		}
		for (int v = 0; v < this.numberOfStates; v++) {
			Arrays.fill(coefficient, 0);
			for (int u = 0; u < this.numberOfStates; u++) {
				coefficient[u * numberOfStates + v] = 1;
			}
			constraints.add(new LinearConstraint(coefficient, Relationship.EQ, this.chain.getProbability(t, v)));
		}

		constraints.add(new LinearConstraint(this.distance, Relationship.LEQ, ACCURACY + this.distance[s * this.numberOfStates + t]));
		
		constraints.add(new LinearConstraint(this.distance, Relationship.GEQ, this.distance[s * this.numberOfStates + t] - ACCURACY));
		

		return constraints;
	}

	/**
	 * Returns an optimal solution for the given states. For states s and t,
	 * getOneOptimalSolution(s, t) == ΛP0(λ0)(s, t).
	 * 
	 * @param s a state
	 * @param t a state
	 * @return ΛP0(λ0)(s, t)
	 */
	public PointValuePair getZeroOptimalSolution(int s, int t) {
		// Objective function
		double[] coefficient = new double[this.numberOfStates * this.numberOfStates];
		for (int v = 0; v < this.numberOfStates; v++) {
			for (int u = 0; u < this.numberOfStates; u++) {
				if (this.bisimilar[u * this.numberOfStates + v]) {
					coefficient[u * numberOfStates + v] = 1;
				} else if (this.differentLabels[u * this.numberOfStates + v]) {
					// Do nothing
				} else {
					coefficient[u * numberOfStates + v] = (1 - this.distance[u * numberOfStates + v])
							+ this.zeroExpectedLengths[u * numberOfStates + v];
				}
			}
		}
		LinearObjectiveFunction objective = new LinearObjectiveFunction(coefficient, 0);

		// Constraints
		Collection<LinearConstraint> constraints = this.getCommonConstraints(s, t);
		Arrays.fill(coefficient, 0);
		for (int v = 0; v < this.numberOfStates; v++) {
			for (int u = 0; u < this.numberOfStates; u++) {
				if (this.bisimilar[u * this.numberOfStates + v]) {
					// do nothing
				} else if (this.differentLabels[u * this.numberOfStates + v]) {
					coefficient[u * numberOfStates + v] = 1;
				} else {
					coefficient[u * numberOfStates + v] = (this.distance[u * numberOfStates + v]
							+ this.oneExpectedLengths[u * numberOfStates + v]);
				}
			}
		}

		constraints.add(new LinearConstraint(coefficient, Relationship.LEQ,
				  this.oneExpectedLengths[s * this.numberOfStates + t]));
		
		constraints.add(new LinearConstraint(coefficient, Relationship.GEQ,
				 this.oneExpectedLengths[s * this.numberOfStates + t] - ACCURACY));

		// Simplex solver
		SimplexSolver solver = new SimplexSolver();
		return solver.optimize(objective, new LinearConstraintSet(constraints), GoalType.MINIMIZE,
				new NonNegativeConstraint(true));
	}

	/**
	 * Initializes the required matrix and vectors for one maximize.
	 */
	public void InitializeOneMatrixVectors() {
		double[] b = new double[this.numberOfStates * this.numberOfStates];
		double[][] A = new double[this.numberOfStates * this.numberOfStates][this.numberOfStates * this.numberOfStates];
		double[] d = new double[this.numberOfStates * this.numberOfStates];

		// Fills the vectors b and d, and the matrix A
		for (int s = 0; s < this.numberOfStates; s++) {
			for (int t = 0; t < this.numberOfStates; t++) {
				if (this.bisimilar[s * this.numberOfStates + t]) {

				} else if (this.differentLabels[s * this.numberOfStates + t]) {

				} else {
					for (int u = 0; u < this.numberOfStates; u++) {
						for (int v = 0; v < this.numberOfStates; v++) {
							if (this.differentLabels[u * this.numberOfStates + v]) {
								b[s * this.numberOfStates
										+ t] += this.policy[s * this.numberOfStates + t][u * this.numberOfStates + v];

							}
							if (!this.bisimilar[u * this.numberOfStates + v]
									&& !this.differentLabels[u * this.numberOfStates + v]) {
								A[s * this.numberOfStates + t][u * this.numberOfStates
										+ v] = this.policy[s * this.numberOfStates + t][u * this.numberOfStates + v];

								d[s * this.numberOfStates + t] = this.distance[s * this.numberOfStates + t];
							}
						}
					}
				}
			}
		}

		// Convert from double to real
		matrixA = MatrixUtils.createRealMatrix(A);

		vectorB = MatrixUtils.createRealVector(b);
		vectorD = MatrixUtils.createRealVector(d);

		// Pre-process
		RealVector vectorT = this.matrixA.operate(this.vectorD);
		this.vectorB = this.vectorB.add(vectorT);

	}

	/**
	 * Computes λ1(s, t) for all states (s, t) in S? using the power method. The
	 * computation result is stored in this.oneExpectedLengths.
	 */
	public void lambdaOnePower() {
		double[] x = new double[this.numberOfStates * this.numberOfStates];
		RealVector vectorX = MatrixUtils.createRealVector(x);

		// Compute λ1
		RealVector vectorT = MatrixUtils.createRealVector(x);

		do {
			vectorT = vectorX.copy();

			vectorX = this.matrixA.operate(vectorX);
			vectorX = vectorX.add(this.vectorB);
		} while (!vectorT.equals(vectorX));

		// Save the results in this.oneExpectedLengths
		for (int s = 0; s < this.numberOfStates; s++) {
			for (int t = 0; t < this.numberOfStates; t++) {
				if (this.bisimilar[s * this.numberOfStates + t]) {

				} else if (this.differentLabels[s * this.numberOfStates + t]) {

				} else {
					this.oneExpectedLengths[(s * this.numberOfStates + t)] = vectorX
							.getEntry(s * this.numberOfStates + t);
				}
			}
		}

	}

	/**
	 * Initializes the required matrix and vectors for zero minimize.
	 */
	public void InitializeZeroMatrixVectors() {

		double[] b = new double[this.numberOfStates * this.numberOfStates];
		double[][] A = new double[this.numberOfStates * this.numberOfStates][this.numberOfStates * this.numberOfStates];
		double[] d = new double[this.numberOfStates * this.numberOfStates];

		// Fills the vectors b and d, and the matrix A
		for (int s = 0; s < this.numberOfStates; s++) {
			for (int t = 0; t < this.numberOfStates; t++) {
				if (this.bisimilar[s * this.numberOfStates + t]) {

				} else if (this.differentLabels[s * this.numberOfStates + t]) {

				} else {
					for (int u = 0; u < this.numberOfStates; u++) {
						for (int v = 0; v < this.numberOfStates; v++) {
							if (this.bisimilar[u * this.numberOfStates + v]) {
								b[s * this.numberOfStates
										+ t] += this.policy[s * this.numberOfStates + t][u * this.numberOfStates + v];

							}
							if (!this.bisimilar[u * this.numberOfStates + v]
									&& !this.differentLabels[u * this.numberOfStates + v]) {
								A[s * this.numberOfStates + t][u * this.numberOfStates
										+ v] = this.policy[s * this.numberOfStates + t][u * this.numberOfStates + v];
								d[s * this.numberOfStates + t] = 1 - this.distance[s * this.numberOfStates + t];
							}
						}
					}
				}
			}
		}

		// Convert from double to real
		this.matrixA = MatrixUtils.createRealMatrix(A);
		this.vectorB = MatrixUtils.createRealVector(b);
		this.vectorD = MatrixUtils.createRealVector(d);

		// Pre-process
		RealVector vectorT = this.matrixA.operate(this.vectorD);
		this.vectorB = vectorB.add(vectorT);

	}
	


	/**
	 * Computes λ0(s, t) for all states (s, t) in S? using the power method. The
	 * computation result is stored in this.zeroExpectedLengths.
	 */
	public void lambdaZeroPower() {
		double[] x = new double[this.numberOfStates * this.numberOfStates];
		RealVector vectorX = MatrixUtils.createRealVector(x);

		// Compute λ0
		RealVector vectorT = MatrixUtils.createRealVector(x);

		do {
			vectorT = vectorX.copy();

			vectorX = matrixA.operate(vectorX);
			vectorX = vectorX.add(this.vectorB);
		} while (!vectorT.equals(vectorX));

		// Save the results in this.oneExpectedLengths
		for (int s = 0; s < this.numberOfStates; s++) {
			for (int t = 0; t < this.numberOfStates; t++) {
				if (this.bisimilar[s * this.numberOfStates + t]) {

				} else if (this.differentLabels[s * this.numberOfStates + t]) {

				} else {
					this.zeroExpectedLengths[(s * this.numberOfStates + t)] = vectorX
							.getEntry(s * this.numberOfStates + t);
				}
			}
		}
	}

	/**
	 * Adds a representation of the computed policy started at the state pair (s, t)
	 * to the given representation. The representation is in DOT format.
	 *
	 * @param s              a state
	 * @param t              a state 
	 * @param representation representation of part of the policy
	 */
	private void policyToDot(int s, int t, StringBuilder representation) {
		String label = String.format("label=\"%d,%d\"", s, t);
		if (!representation.toString().contains(label)) {
			String shape;
			String color;
			if (this.bisimilar[s * this.numberOfStates + t]) {
				shape = "diamond";
				color = "blue";
			} else if (this.differentLabels[s * this.numberOfStates + t]) {
				shape = "square";
				color = "red";
			} else {
				shape = "circle";
				color = "green";
			}
			representation.append(
					String.format("%d [shape=%s, color=%s, %s]\n", s * this.numberOfStates + t, shape, color, label));

			double[] coupling = this.policy[s * this.numberOfStates + t];
			for (int u = 0; u < this.numberOfStates; u++) {
				for (int v = 0; v < this.numberOfStates; v++) {
					double value = coupling[u * this.numberOfStates + v];
					if (value > 0) {
						if (u > v) {
							this.policyToDot(v, u, representation);
							representation.append(String.format(
									"%d -> %d [label=\"%." + Constants.PRECISION + "f\", arrowhead=normalicurve]\n",
									s * this.numberOfStates + t, v * this.numberOfStates + u, value));
						} else {
							this.policyToDot(u, v, representation);
							representation.append(String.format("%d -> %d [label=\"%." + Constants.PRECISION + "f\"]\n",
									s * this.numberOfStates + t, u * this.numberOfStates + v, value));
						}
					}
				}
			}
		}
	}

	/**
	 * Returns a representation of the computed policy started at the state pair (s,
	 * t). The representation is in DOT format.
	 *
	 * @param s a state
	 * @param t a state
	 * @return a representation of the computed policy started at the state pair (s,
	 *         t)
	 */
	public String policyToDot(int s, int t) {
		StringBuilder representation = new StringBuilder();
		representation.append("digraph Policy {\n");
		if (s > t) {
			this.policyToDot(t, s, representation);
		} else {
			this.policyToDot(s, t, representation);
		}
		representation.append("}\n");
		return representation.toString();
	}

}
