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

import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.Set;

import explainability.lmc.LabelledMarkovChain;

/**
 * Decides which states of a labelled Markov chain have probabilistic bisimilarity 
 * distance zero, that is, they are probabilistic bisimilar.  The implementation is 
 * based on the algorithm from the paper "Optimal State-Space Lumping in Markov Chains" 
 * by Salem Derisavi, Holger Hermanns, and William Sanders.
 * 
 * @author Zainab Fatmi
 */
public class Derisavi {
	
	/**
	 * A class to represent the nodes of a splay tree.  Each node of the tree stores 
	 * a block and its probability of transitioning to the current splitter.
	 */
	private static class Node {
		private Block block;
		private double probability;
		private Node parent;
		private Node left; // left child
		private Node right; // right child

		/**
		 * Initializes this node with an empty block, the given probability and the given parent node.
		 * 
		 * @param probability the probability of the states in the block of this node transitioning to the 
		 * current splitter
		 * @param parent the parent node
		 */
		public Node(double probability, Node parent) {
			this.block = new Block();
			this.probability = probability;
			this.parent = parent;
			this.left = null;
			this.right = null;
		}
	}
	
	/**
	 * A splay tree.  Each node of the tree stores a block and its probability of 
	 * transitioning to the current splitter.
	 */
	private static class SplayTree {
		private Node root;

		/**
		 * Initializes this splay tree as empty.
		 */
		public SplayTree() {
			this.root = null;
		}

		/**
		 * Inserts the state in its appropriate position in this splay tree. If the probability
		 * exists in the splay tree, adds the state to the block associated to the probability,
		 * otherwise creates a new node in this splay tree.
		 * 
		 * @param probability the probability of the state transitioning to the current splitter
		 * @param state a state
		 */
		public void insert(double probability, State state) {
			Node cursor = this.root;
			Node parent = null; // the parent of cursor
			while (cursor != null && Math.abs(probability - cursor.probability) >= ACCURACY) {
				parent = cursor;
				if (probability < cursor.probability) {
					cursor = cursor.left;
				} else {
					cursor = cursor.right;
				}
			}
			if (cursor == null) {
				Node node = new Node(probability, parent);
				if (parent == null) {
					root = node;
				} else if (probability < parent.probability) {
					parent.left = node;
				} else {
					parent.right = node;
				}
				node.block.elements.add(state);
				state.block = node.block;
				splay(node);
			} else {
				cursor.block.elements.add(state);
				state.block = cursor.block;
				splay(cursor);
			}
		}

		/**
		 * Moves the given node to the root of the splay tree.
		 * 
		 * @param node a node
		 */
		private void splay(Node node) {
			while (node.parent != null) {
				if (node.parent.parent == null) {
					if (node == node.parent.left) {
						// zig rotation
						this.rotateRight(node.parent);
					} else {
						// zag rotation
						this.rotateLeft(node.parent);
					}
				} else if (node == node.parent.left && node.parent == node.parent.parent.left) {
					// zig-zig rotation
					this.rotateRight(node.parent.parent);
					this.rotateRight(node.parent);
				} else if (node == node.parent.right && node.parent == node.parent.parent.right) {
					// zag-zag rotation
					this.rotateLeft(node.parent.parent);
					this.rotateLeft(node.parent);
				} else if (node == node.parent.right && node.parent == node.parent.parent.left) {
					// zig-zag rotation
					this.rotateLeft(node.parent);
					this.rotateRight(node.parent);
				} else {
					// zag-zig rotation
					this.rotateRight(node.parent);
					this.rotateLeft(node.parent);
				}
			}
		}

		/**
		 * Rotates left at the given node.
		 * 
		 * @param node a node
		 */
		private void rotateLeft(Node node) {
			Node child = node.right;
			node.right = child.left;
			if (node.right != null) {
				node.right.parent = node;
			}
			child.parent = node.parent;
			if (node.parent == null) {
				this.root = child;
			} else if (node == node.parent.left) {
				node.parent.left = child;
			} else {
				node.parent.right = child;
			}
			child.left = node;
			node.parent = child;
		}

		/**
		 * Rotates right at the given node.
		 * 
		 * @param node a node
		 */
		private void rotateRight(Node node) {
			Node child = node.left;
			node.left = child.right;
			if (node.left != null) {
				node.left.parent = node;
			}
			child.parent = node.parent;
			if (node.parent == null) {
				this.root = child;
			} else if (node == node.parent.right) {
				node.parent.right = child;
			} else {
				node.parent.left = child;
			}
			child.right = node;
			node.parent = child;
		}
	}

	/**
	 * A class to represent the blocks of the partition.
	 */
	private static class Block {
		private static int numberOfBlocks = 0;
		
		private int id; // for easier hashCode and equals methods
		private LinkedList<State> elements;
		private SplayTree tree;

		/**
		 * Initializes this block as empty and adds it to the partition.
		 */
		public Block() {
			this.id = numberOfBlocks++;
			this.elements = new LinkedList<State>();
			this.tree = new SplayTree();
			partition.add(this);
		}

		@Override
		public int hashCode() {
			return this.id;
		}

		@Override
		public boolean equals(Object object) {
			if (this != null && this.getClass() == object.getClass()) {
				Block other = (Block) object;
				return this.id == other.id;
			} else {
				return false;
			}
		}
	}

	/**
	 * A class to represent the states of the labelled Markov chain.
	 */
	private static class State {
		private int id;
		private Block block; // needed by the splay tree
		private double sum;
		private LinkedHashMap<State, Double> predecessors; // no need for successors

		/**
		 * Initializes this state with the given index.
		 * 
		 * @param id the non-negative ID of the state
		 */
		public State(int id) {
			this.id = id;
			this.sum = 0;
			this.predecessors = new LinkedHashMap<State, Double>();
		}

		@Override
		public int hashCode() {
			return this.id;
		}

		@Override
		public boolean equals(Object object) {
			if (this != null && this.getClass() == object.getClass()) {
				State other = (State) object;
				return this.id == other.id;
			} else {
				return false;
			}
		}
	}

	/**
	 * Partition of the states into blocks.
	 */
	private static LinkedList<Block> partition;
	
	/**
	 * Decides probabilistic bisimilarity distance zero for the given labelled Markov chain.
	 * 
	 * @param chain a labelled Markov chain
	 * @return a boolean array that captures for each state pair whether
	 * the states have probabilistic bisimilarity distance zero:
	 * zero[s * chain.getNumberOfStates() + t] == states s and t have distance zero
	 */
	public static boolean[] decide(LabelledMarkovChain chain) {
		int numberOfStates = chain.getNumberOfStates();
		int numberOfLabels = chain.getNumberOfLabels();
		
		// start with an empty partition
		partition = new LinkedList<Block>();
		
		// create an empty block for each label and add it to the partition
		for (int i = 0; i < numberOfLabels; i++) {
			new Block();
		}
		
		// add the states to the blocks corresponding to the label of the state 
		State[] idToState = new State[numberOfStates]; // map id to State
		for (int id = 0; id < numberOfStates; id++) {
			State state = new State(id);
			idToState[id] = state;
			Block block = partition.get(chain.getLabel(id));
			block.elements.add(state);
			state.block = block;
		}
		for (int source = 0; source < numberOfStates; source++) {
			for (int target = 0; target < numberOfStates; target++) {
				if (chain.getProbability(source, target) != 0.0) {
					idToState[target].predecessors.put(idToState[source], chain.getProbability(source, target));
				}
			}
		}
		
		LinkedList<Block> potentialSplitters = new LinkedList<Block>(partition); // potential splitters
		Set<State> predecessors = new HashSet<State>(); // states that have a transition to the current splitter
		LinkedList<Block> partitioned = new LinkedList<Block>(); // blocks which will be partitioned

		while (!potentialSplitters.isEmpty()) {
			Block splitter = potentialSplitters.pop();

			predecessors.clear();
			for (State state : splitter.elements) {
				for (State predecessor : state.predecessors.keySet()) {
					predecessor.sum = 0;
				}
			}
			for (State state : splitter.elements) {
				for (Map.Entry<State, Double> entry : state.predecessors.entrySet()) {
					State predecessor = entry.getKey();
					predecessor.sum += entry.getValue();
					predecessors.add(predecessor);
				}
			}
			
			partitioned.clear();
			for (State state : predecessors) {
				Block block = state.block;
				block.elements.remove(state);
				block.tree.insert(state.sum, state);
				if (!partitioned.contains(block)) {
					partitioned.add(block);
				}
			}
			
			for (Block block : partitioned) {
				// traverse the subblock tree, adding subblocks and keeping track of the maximum
				Block max = block;
				LinkedList<Node> queue = new LinkedList<Node>();
				queue.add(block.tree.root);
				while (!queue.isEmpty()) {
					Node node = queue.removeFirst();
					if (node.left != null) {
						queue.add(node.left);
					}
					if (node.right != null) {
						queue.add(node.right);
					}
					if (node.block.elements.size() > max.elements.size()) {
						max = node.block;
					}
					potentialSplitters.add(node.block);
				}
				
				if (!potentialSplitters.contains(block) && !(max == block)) {
					potentialSplitters.add(block);
					potentialSplitters.remove(max);
				}

				if (block.elements.isEmpty()) {
					partition.remove(block);
					potentialSplitters.remove(block);
				} else {
					block.tree.root = null; // reset the splay tree
				}
			}
		}
		
		boolean[] zero = new boolean[numberOfStates * numberOfStates];
		for (Block block : partition) {
			for (State s : block.elements) {
				for (State t : block.elements) {
					assert t.id < numberOfStates;
					zero[s.id * numberOfStates + t.id] = true;
				}
			}
		}
		return zero;
	}
}
