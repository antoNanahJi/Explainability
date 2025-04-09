/*
 * Copyright (C)  2018  Qiyi Tang
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

package explainability.lmc.compute;

import static explainability.lmc.Constants.ACCURACY;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.linear.LinearConstraint;
import org.apache.commons.math3.optim.linear.LinearConstraintSet;
import org.apache.commons.math3.optim.linear.LinearObjectiveFunction;
import org.apache.commons.math3.optim.linear.NonNegativeConstraint;
import org.apache.commons.math3.optim.linear.Relationship;
import org.apache.commons.math3.optim.linear.SimplexSolver;
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;

import explainability.lmc.LabelledMarkovChain;
import explainability.lmc.decide.Derisavi;

/**
 * Computes the probabilistic bisimilarity distances by
 * means of simple policy iteration.
 * 
 * @author Qiyi Tang
 */
public class SimplePolicyIteration {

	/*
	 * Labelled Markov chain of which the distances are computed.
	 */
	private LabelledMarkovChain chain;

	/*
	 * The number of states of the labelled Markov chain.
	 */
	private final int numberOfStates;

	/*
	 * The distances of the labelled Markov chain:
	 * distance[s * numberOfStates + t] = distance of s and t.
	 */
	private double[] distance;

	/*
	 * For states s, t, with s > t,
	 * toCompute[s * numberOfStates + t] == whether the distance 
	 * of s and t still needs to be computed. 
	 */
	private boolean[] toCompute;

	/*
	 * For states s, t, with s > t, if 
	 * toCompute[s * numberOfStates + t]
	 * then policy[s * numberOfStates + t] is a coupling of
	 * chain.getProbability(s) and chain.getProbability(t).
	 */
	private double[][] policy;
	
	/*
	 * For states s, t,
	 * zero[s * numberOfStates + t] == distance of s and t is zero.
	 */
	private boolean[] bisimilar;
	
	/*
	 * For states s, t,
	 * differentLabels[s * numberOfStates + t] == s and t have different labels.
	 */
	private boolean[] differentLabels;

	/**
	 * Initializes this computation of probabilistic bisimilarity
	 * distances for the given labelled Markov chain.
	 * 
	 * @param chain a labelled Markov chain
	 */
	public SimplePolicyIteration(LabelledMarkovChain chain) {
		this.chain = chain;
		this.numberOfStates = chain.getNumberOfStates();
		this.distance = new double[numberOfStates * numberOfStates];
		this.toCompute = new boolean[numberOfStates * numberOfStates];
		this.policy = new double[numberOfStates * numberOfStates][numberOfStates * numberOfStates];
		this.bisimilar = Derisavi.decide(chain);
		this.differentLabels = new boolean[numberOfStates * numberOfStates];
		
		for (int s = 0; s < this.numberOfStates; s++) {
			for (int t = 0; t < this.numberOfStates; t++) {
				if (this.bisimilar[s * this.numberOfStates + t]) {
					// do nothing
				} else if (chain.getLabel(s) != chain.getLabel(t)) {
					this.distance[s * this.numberOfStates + t] = 1;
					this.differentLabels[s * this.numberOfStates + t] = true;
				} else {
					this.toCompute[s * this.numberOfStates + t] = true;
					this.setInitialCoupling(s, t);
				}
			}
		}
	}

	/**
	 * Sets a coupling for the probability distributions
	 * of the given states.  
	 * The coupling is computed using the North-West corner method.
	 * 
	 * @param s a state
	 * @param t a state
	 * @pre. s > t
	 */
	public void setInitialCoupling(int s, int t) {
		double[] source = Arrays.copyOf(this.chain.getProbability(s), this.numberOfStates);
		double[] target = Arrays.copyOf(this.chain.getProbability(t), this.numberOfStates);
	
		for (int u = 0; u < this.numberOfStates; u++) {
			for (int v = 0; v < this.numberOfStates; v++) {
				double minimum = Math.min(source[u], target[v]);
				this.policy[s * this.numberOfStates + t][u * this.numberOfStates + v] = minimum;
				source[u] -= minimum;
				target[v] -= minimum;
			}
		}
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
	 * Returns whether states are probabilistic bisimilar.
	 * For states s, t,
	 * getBisimilar()[s * chain.getNumberOfStates() + t] == s and t are probabilistic bisimilar.
	 * 
	 * @return Returns whether states are probabilistic bisimilar.
	 */
	public boolean[] getBisimilar() {
		return this.bisimilar;
	}
	
	/**
	 * Returns whether states have different labels.
	 * For states s, t,
	 * getDifferentLabels()[s * chain.getNumberOfStates() + t] == s and t have different labels.
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
	 * Sets the distances based on the current couplings.
	 */
	public void setDistance() {
		double[] lower = new double[this.numberOfStates * this.numberOfStates];
		double[] upper = new double[this.numberOfStates * this.numberOfStates];
		
		for (int u = 0; u < this.numberOfStates; u++) {
			for (int v = 0; v < this.numberOfStates; v++) {
				if (this.bisimilar[u * this.numberOfStates + v]) {
					// lower[u * this.numberOfStates + v] = 0;
					// upper[u * this.numberOfStates + v] = 0;
				} else if (this.chain.getLabel(u) != this.chain.getLabel(v)) {
					lower[u * this.numberOfStates + v] = 1;
					upper[u * this.numberOfStates + v] = 1;
				} else {
					// lower[u * this.numberOfStates + v] = 0;
					upper[u * this.numberOfStates + v] = 1;
				}
			}		
		}
		
		boolean[] farApart = Arrays.copyOf(this.toCompute, this.toCompute.length);
		boolean done;
		do {
			done = true;
			double[] previousLower = Arrays.copyOf(lower, lower.length);
			double[] previousUpper = Arrays.copyOf(upper, upper.length);
			for (int s = 0; s < this.numberOfStates; s++) {
				for (int t = 0; t < this.numberOfStates; t++) {
					if (farApart[s * this.numberOfStates + t]) {
						lower[s * this.numberOfStates + t] = 0;
						upper[s * this.numberOfStates + t] = 0;
						for (int u = 0; u < this.numberOfStates; u++) {
							for (int v = 0; v < this.numberOfStates; v++) {
								if (this.bisimilar[u * this.numberOfStates + v]) {
									// do nothing
								} else {
									double value = this.policy[s * this.numberOfStates + t][u * this.numberOfStates + v];	
									if (this.chain.getLabel(u) != this.chain.getLabel(v)) {
										lower[s * this.numberOfStates + t] += value;
										upper[s * this.numberOfStates + t] += value;
									} else { 
										lower[s * this.numberOfStates + t] += previousLower[u * this.numberOfStates + v] * value;
										upper[s * this.numberOfStates + t] += previousUpper[u * this.numberOfStates + v] * value;
									}
								}
							}
						}

						if (upper[s * this.numberOfStates + t] - lower[s * this.numberOfStates + t] < ACCURACY) {
							farApart[s * this.numberOfStates + t] = false;
						} else {
							done = false;
						}
					}
				}
			}
		} while (!done);

		for (int s = 0; s < this.numberOfStates; s++) {
			for (int t = 0; t < this.numberOfStates; t++) {
				if (this.toCompute[s * this.numberOfStates + t]) {
					this.distance[s * this.numberOfStates + t] = (lower[s * this.numberOfStates + t] + upper[s * this.numberOfStates + t]) / 2;
				}
			}
		}
	}

	/**
	 * Tests whether the coupling for the given states is optimal.
	 * If not, updates the coupling for the given states.
	 * 
	 * @param s a state
	 * @param t a state
	 * @pre. s > t
	 * @return true if the coupling for the given states is optimal,
	 * false otherwise
	 */
	public boolean isOptimalCoupling(int s, int t) {
		PointValuePair solution = this.getOptimalSolution(s, t);
		double value = solution.getValue();
		if (value + ACCURACY >= this.distance[s * this.numberOfStates + t]) {
			return true;
		} else {
			double[] point = solution.getPoint();
			for (int u = 0; u < this.numberOfStates; u++) {
				for (int v = 0; v < this.numberOfStates; v++) {
					if (this.toCompute[u * this.numberOfStates + v]) {
						this.policy[s * this.numberOfStates + t] = point;
					}
				}
			}
			return false;
		}
	}

	/**
	 * Returns an optimal solution for the given states.
	 * 
	 * @param s a state
	 * @param t a state
	 * @pre. s > t
	 * @return an optimal solution for the given states
	 */
	public PointValuePair getOptimalSolution(int s, int t) {
		// objective function
		LinearObjectiveFunction objective = new LinearObjectiveFunction(this.distance, 0);

		// constraints
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

		SimplexSolver solver = new SimplexSolver();
		return solver.optimize(objective, new LinearConstraintSet(constraints), GoalType.MINIMIZE, new NonNegativeConstraint(true));
	}

	/**
	 * Returns the probabilistic bisimilarity distances for the
	 * given labelled Markov chain.
	 * 
	 * @param chain a labelled Markov chain
	 * @return the probabilistic bisimilarity distances for the
	 * given labelled Markov chain
	 */
	public void compute() {
		this.setDistance();
		boolean allOptimal;
		do {
			allOptimal = true;
			for (int s = 0; s < this.numberOfStates && allOptimal; s++) {
				for (int t = 0; t < this.numberOfStates && allOptimal; t++) {
					if (this.toCompute[s * this.numberOfStates + t] && !this.isOptimalCoupling(s, t)) {
						allOptimal = false;
						this.setDistance();
					}
				}
			}
		} while (!allOptimal);
	}
}