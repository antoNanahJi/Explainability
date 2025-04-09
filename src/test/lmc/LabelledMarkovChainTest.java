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

package test.lmc;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Random;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.RepeatedTest;

import explainability.lmc.LabelledMarkovChain;

/**
 * Tests the LabelledMarkovChain class.
 * 
 * @author Franck van Breugel
 */
class LabelledMarkovChainTest {

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
	 * Tests the constructor, getNumberOfStates, getLabels, getProbabilities, and equals methods.
	 */
	@RepeatedTest(TIMES)
	void testConstructor() {
		int numberOfStates = 1 + RANDOM.nextInt(MAX_STATES);
		int maximalNumberOfLabels = 1 + RANDOM.nextInt(MAX_LABELS);
		LabelledMarkovChain expected = LabelledMarkovChain.randomUniform(numberOfStates, maximalNumberOfLabels);
		
		Assertions.assertEquals(numberOfStates, expected.getNumberOfStates());
		
		try {
			PrintWriter output = new PrintWriter("temp.lab");
			output.print(expected.getLabels());
			output.close();
		} catch (FileNotFoundException e) {
			Assertions.fail("Could not write to the file temp.lab");
		}

		try {
			PrintWriter output = new PrintWriter("temp.tra");
			output.println(numberOfStates + " " + numberOfStates * numberOfStates);
			output.print(expected.getProbabilities());
			output.close();
		} catch (FileNotFoundException e) {
			Assertions.fail("Could not write to the file temp.tra");
		}
		
		try {
			LabelledMarkovChain actual = new LabelledMarkovChain("temp");
			Assertions.assertEquals(expected, actual);
			File file = new File("temp.lab");
			file.delete();
			file = new File("temp.tra");
			file.delete();
		} catch (IllegalArgumentException e) {
			Assertions.fail(e.getMessage());
		} catch (FileNotFoundException e) {
			Assertions.fail(e.getMessage());
		}
	}
	
	/**
	 * Tests the getNumberOfStates method.
	 */
	@RepeatedTest(TIMES)
	void testGetNumberOfStates() {
		int numberOfStates = 1 + RANDOM.nextInt(MAX_STATES);
		int maximalNumberOfLabels = 1 + RANDOM.nextInt(MAX_LABELS);
		LabelledMarkovChain chain = LabelledMarkovChain.randomUniform(numberOfStates, maximalNumberOfLabels);
	
		Assertions.assertEquals(numberOfStates, chain.getNumberOfStates());
	}
	
	/**
	 * Tests the getNumberOfLabels method.
	 */
	@RepeatedTest(TIMES)
	void testGetNumberOfLabels() {
		int numberOfStates = 1 + RANDOM.nextInt(MAX_STATES);
		int maximalNumberOfLabels = 1 + RANDOM.nextInt(MAX_LABELS);
		LabelledMarkovChain chain = LabelledMarkovChain.randomUniform(numberOfStates, maximalNumberOfLabels);
	
		Assertions.assertTrue(chain.getNumberOfLabels() <= maximalNumberOfLabels);
	}
}
