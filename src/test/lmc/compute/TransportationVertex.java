/*
 * Copyright (C)  2020  Amgad Rady and Franck van Breugel
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

package test.lmc.compute;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.linear.LinearConstraint;
import org.apache.commons.math3.optim.linear.LinearConstraintSet;
import org.apache.commons.math3.optim.linear.LinearObjectiveFunction;
import org.apache.commons.math3.optim.linear.NonNegativeConstraint;
import org.apache.commons.math3.optim.linear.Relationship;
import org.apache.commons.math3.optim.linear.SimplexSolver;
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;

/**
 * Finds a vertex of the transportation polytope.
 * 
 * @author Amgad Rady
 * @author Franck van Breugel
 */
public class TransportationVertex {
	
	/**
	 * Returns a vertex and the corresponding value of the transportation polytope.
	 * 
	 * @param distance the distances of the states of the labelled Markov chain
	 * @param first the transition probabilities of a state of the labelled Markov chain
	 * @param second the transition probabilities of a state of the labelled Markov chain
	 * @return a vertex and the corresponding value of the transportation polytope
	 */
	public static PointValuePair find(double[] distance, double[] first, double[] second) {
		int numberOfStates = first.length;

		// objective function
		LinearObjectiveFunction objectiveFunction = new LinearObjectiveFunction(distance, 0.0);

		// constraints
		Set<LinearConstraint> constraintSet = new HashSet<LinearConstraint>();
		double[] coefficient = new double[numberOfStates * numberOfStates];
		for (int v = 0; v < numberOfStates; v++) {
			/*
			 * For all 0 <= v < numberOfStates, sum 0 <= u < numberOfStates, coupling[u * numberOfStates + v] = second[v]
			 */
			Arrays.fill(coefficient, 0);
			for (int u = 0; u < numberOfStates; u++) {
				coefficient[u * numberOfStates + v] = 1;
			}
			constraintSet.add(new LinearConstraint(coefficient, Relationship.EQ, second[v]));
		}
		for (int u = 0; u < numberOfStates; u++) {
			/*
			 * For all 0 <= u < numberOfStates, sum 0 <= v < numberOfStates, coupling[u * numberOfStates + v] = first[u]
			 */
			Arrays.fill(coefficient, 0);
			for (int v = 0; v < numberOfStates; v++) {
				coefficient[u * numberOfStates + v] = 1;
			}
			constraintSet.add(new LinearConstraint(coefficient, Relationship.EQ, first[u]));
		}
		LinearConstraintSet constraints = new LinearConstraintSet(constraintSet);

		// solve
		SimplexSolver solver = new SimplexSolver();
		return solver.optimize(objectiveFunction, constraints, new NonNegativeConstraint(true), GoalType.MINIMIZE);
	}
}
