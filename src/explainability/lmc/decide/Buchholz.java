/*
 * Copyright (C)  2020  Zainab Fatmi
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

package explainability.lmc.decide;

import static explainability.lmc.Constants.ACCURACY;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import explainability.lmc.LabelledMarkovChain;

/**
 * Decides which states of a labelled Markov chain have probabilistic bisimilarity 
 * distance zero, that is, they are probabilistic bisimilar.  The implementation
 * is based on the bisimilarity algorithm from the paper "Efficient computation of 
 * equivalent and reduced representations for stochastic automata" by Peter Buchholz.
 * 
 * @author Zainab Fatmi
 */
public class Buchholz {
	
	/**
	 * A class to represent the equivalence classes during the split method.
	 */
	private static class EquivalenceClass {
		private boolean initialized;
		private double value;
		private int next;

		/**
		 * Initializes this equivalence class as uninitialized (no state belonging to this equivalence
		 * class has been found yet).
		 */
		public EquivalenceClass() {
			this.initialized = false;
			this.value = 0;
			this.next = 0;
		}
	}

	/**
	 * Decides probabilistic bisimilarity distance zero for the given labelled Markov chain.
	 * 
	 * @param chain a labelled Markov chain
	 * @return a boolean array that captures for each state pair whether
	 * the states have probabilistic bisimilarity distance zero:
	 * zero[s * chain.getNumberOfStates() + t] == states s and t have distance zero
	 */
	public static boolean[] decide(LabelledMarkovChain chain) {
		// assign each label an index
		List<Integer> indices = new ArrayList<Integer>();
		for (int state = 0; state < chain.getNumberOfStates(); state++) {
			int label = chain.getLabel(state);
			if (!indices.contains(label)) {
				indices.add(label);
			}
		}
		
		// partition by labels
		int numberOfEquivalenceClasses = indices.size(); // number of equivalence classes
		List<Set<Integer>> classes = new ArrayList<Set<Integer>>(); // equivalence classes
		TreeSet<Integer> splitters = new TreeSet<Integer>(); // potential splitters
		int[] clazzOf = new int[chain.getNumberOfStates()]; // for each state ID, the index of its equivalence class
		for (int clazz = 0; clazz < numberOfEquivalenceClasses; clazz++) {
			classes.add(new HashSet<Integer>());
			splitters.add(clazz);
		}
		for (int state = 0; state < chain.getNumberOfStates(); state++) {
			int label = chain.getLabel(state);
			int index = indices.indexOf(label);
			clazzOf[state] = index;
			classes.get(index).add(state);
		}
		
		double[] values = new double[chain.getNumberOfStates()];
		while (!splitters.isEmpty()) {
			List<EquivalenceClass> split = new ArrayList<EquivalenceClass>();
			for (int clazz = 0; clazz < numberOfEquivalenceClasses; clazz++) {
				split.add(new EquivalenceClass());
			}
			int splitter = splitters.first();
			splitters.remove(splitter);
			Arrays.fill(values, 0);
			for (int target : classes.get(splitter)) {
				for (int source = 0; source < chain.getNumberOfStates(); source++) {
					values[source] += chain.getProbability(source, target);
				}
			}
			
			for (int state = 0; state < chain.getNumberOfStates(); state++) {
				int clazz = clazzOf[state];
				if (!split.get(clazz).initialized) {
					classes.set(clazz, new HashSet<Integer>());
					classes.get(clazz).add(state);
					split.get(clazz).initialized = true;
					split.get(clazz).value = values[state];
				} else {
					if (Math.abs(split.get(clazz).value - values[state]) >= ACCURACY && split.get(clazz).next == 0) {
						splitters.add(clazz);
					}
					while (Math.abs(split.get(clazz).value - values[state]) >= ACCURACY && split.get(clazz).next != 0) {
						clazz = split.get(clazz).next;
					}
					if (Math.abs(split.get(clazz).value - values[state]) < ACCURACY) {
						clazzOf[state] = clazz;
						classes.get(clazz).add(state);
					} else {
						splitters.add(numberOfEquivalenceClasses);
						clazzOf[state] = numberOfEquivalenceClasses;
						split.get(clazz).next = numberOfEquivalenceClasses;
						split.add(new EquivalenceClass());
						split.get(numberOfEquivalenceClasses).initialized = true;
						split.get(numberOfEquivalenceClasses).value = values[state];
						classes.add(new HashSet<Integer>());
						classes.get(numberOfEquivalenceClasses).add(state);
						numberOfEquivalenceClasses++;
					}
				}
			}
		}
		
		boolean[] zero = new boolean[chain.getNumberOfStates() * chain.getNumberOfStates()];
		for (Set<Integer> clazz : classes) {
			for (Integer s : clazz) {
				for (Integer t : clazz) {
					zero[s * chain.getNumberOfStates() + t] = true;
				}
			}
		}
		return zero;
	}
}
