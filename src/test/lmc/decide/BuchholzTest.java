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

package test.lmc.decide;

import java.util.Random;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.RepeatedTest;

import explainability.lmc.LabelledMarkovChain;
import explainability.lmc.decide.Buchholz;

/**
 * Tests the Buchholz class.
 * 
 * @author Franck van Breugel
 */
class BuchholzTest {

	/**
	 * Randomness.
	 */
	private static final Random RANDOM = new Random();

	/**
	 * Number of times that random tests are repeated.
	 */
	private static final int TIMES = 1000;

	/**
	 * Maximum number of states of a random labelled Markov chain.
	 */
	private static final int MAX_STATES = 100;

	/**
	 * Maximum number of labels of a random labelled Markov chain.
	 */
	private static final int MAX_LABELS = 100;

	/**
	 * Tests that probabilistic bisimilarity is reflexive.
	 */
	@RepeatedTest(TIMES)
	void testReflective() {
		int numberOfStates = 1 + RANDOM.nextInt(MAX_STATES);
		int maximalNumberOfLabels = 1 + RANDOM.nextInt(MAX_LABELS);
		LabelledMarkovChain chain = LabelledMarkovChain.randomUniform(numberOfStates, maximalNumberOfLabels);

		boolean[] bisimilar = Buchholz.decide(chain);
		for (int state = 0; state < numberOfStates; state++) {
			Assertions.assertTrue(bisimilar[state * numberOfStates + state]);
		}
	}

	/**
	 * Tests that probabilistic bisimilarity is symmetric.
	 */
	@RepeatedTest(TIMES)
	void testSymmetric() {
		int numberOfStates = 1 + RANDOM.nextInt(MAX_STATES);
		int maximalNumberOfLabels = 1 + RANDOM.nextInt(MAX_LABELS);
		LabelledMarkovChain chain = LabelledMarkovChain.randomUniform(numberOfStates, maximalNumberOfLabels);

		boolean[] bisimilar = Buchholz.decide(chain);
		for (int s = 0; s < numberOfStates; s++) {
			for (int t = 0; t < numberOfStates; t++) {
				Assertions.assertEquals(bisimilar[s * numberOfStates + t], bisimilar[t * numberOfStates + s]);
			}
		}
	}

	/**
	 * Tests that probabilistic bisimilarity is transitive.
	 */
	@RepeatedTest(TIMES)
	void testTransitive() {
		int numberOfStates = 1 + RANDOM.nextInt(MAX_STATES);
		int maximalNumberOfLabels = 1 + RANDOM.nextInt(MAX_LABELS);
		LabelledMarkovChain chain = LabelledMarkovChain.randomUniform(numberOfStates, maximalNumberOfLabels);

		boolean[] bisimilar = Buchholz.decide(chain);
		for (int s = 0; s < numberOfStates; s++) {
			for (int t = 0; t < numberOfStates; t++) {
				for (int u = 0; u < numberOfStates; u++) {
					if (bisimilar[s * numberOfStates + t] && bisimilar[t * numberOfStates + u]) {
						Assertions.assertTrue(bisimilar[s * numberOfStates + u]);
					}
				}
			}
		}
	}

	/**
	 * Tests that all states are probabilistic bisimilar if there is only one label.
	 */
	@RepeatedTest(TIMES)
	void testOneLabel() {
		int numberOfStates = 1 + RANDOM.nextInt(MAX_STATES);
		int maximalNumberOfLabels = 1;
		LabelledMarkovChain chain = LabelledMarkovChain.randomUniform(numberOfStates, maximalNumberOfLabels);

		boolean[] bisimilar = Buchholz.decide(chain);
		for (int s = 0; s < numberOfStates; s++) {
			for (int t = 0; t < numberOfStates; t++) {
				Assertions.assertTrue(bisimilar[s * numberOfStates + t]);
			}
		}
	}
}
