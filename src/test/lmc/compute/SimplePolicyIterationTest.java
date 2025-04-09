/*
 * Copyright (C)  2022  Franck van Breugel
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

import static explainability.lmc.Constants.ACCURACY;

import java.util.Random;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.RepeatedTest;

import explainability.lmc.LabelledMarkovChain;
//import explainability.lmc.compute.IncorrectSimplePolicyIteration;
import explainability.lmc.compute.SimplePolicyIteration;
import explainability.lmc.decide.Derisavi;

/**
 * Tests the computation of the probabilistic bisimilarity distances
 * by means of simply policy iteration.
 * 
 * @author Franck van Breugel
 */
public class SimplePolicyIterationTest {

	/**
	 * Randomness.
	 */
	private static final Random random = new Random();

	/**
	 * Number of times that tests with randomness are run.
	 */
	private static final int TIMES = 100;

	/**
	 * Maximal number of states of a labelled Markov chain.
	 */
	private static final int MAX_STATES = 20;

	/**
	 * Maximal number of labels of a labelled Markov chain.
	 */
	private static final int MAX_LABELS = 5;

	/**
	 * Tests the computation of the probabilistic bisimilarity distances
	 * by means of simply policy iteration.
	 */
	@RepeatedTest(TIMES)
	void testCompute() {
		int numberOfStates = 1 + random.nextInt(MAX_STATES);
		int numberOfLabels = 1 + random.nextInt(MAX_LABELS);

		LabelledMarkovChain chain = LabelledMarkovChain.randomUniform(numberOfStates, numberOfLabels);
		SimplePolicyIteration iteration = new SimplePolicyIteration(chain);
		iteration.compute();
		
		double[] distance = iteration.getDistance(); 
		boolean[] bisimilar = Derisavi.decide(chain);

		for (int s = 0; s < numberOfStates; s++) {
			for (int t = 0; t < s; t++) {
				Assertions.assertEquals(bisimilar[s * numberOfStates + t], distance[s * numberOfStates + t] == 0, String.format("Distance for (%d, %d) was expected to be %s zero but was %f\n", s, t, bisimilar[s * numberOfStates + t] ? "" : "not", distance[s * numberOfStates + t])); 
				Assertions.assertTrue(chain.getLabel(s) == chain.getLabel(t) || distance[s * numberOfStates + t] == 1, String.format("Distance for (%d, %d) was expected to be one but was %f\n", s, t, distance[s * numberOfStates + t])); 
				if (!bisimilar[s * numberOfStates + t] && chain.getLabel(s) == chain.getLabel(t)) {
					double expected = LipschitzVertex.find(distance, chain.getProbability(s), chain.getProbability(t)).getValue();
					Assertions.assertEquals(expected, distance[s * numberOfStates + t], ACCURACY,
							String.format("Labelled Markov chain\n%s\nDistance for (%d, %d)\n", chain, s, t));
					expected = TransportationVertex.find(distance, chain.getProbability(s), chain.getProbability(t)).getValue();
					Assertions.assertEquals(expected, distance[s * numberOfStates + t], ACCURACY,
							String.format("Labelled Markov chain\n%s\nDistance for (%d, %d)\n", chain, s, t));
				}
			}
		}
	}
}
