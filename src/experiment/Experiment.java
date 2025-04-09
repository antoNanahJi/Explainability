package experiment;

import java.io.FileNotFoundException;

import explainability.lmc.Explanation;
import explainability.lmc.LabelledMarkovChain;

/**
 * Performs experiments on policies that explain the probabilistic bisimilarity
 * distances of a labelled Markov chain.
 *
 * @author Anto Nanah Ji
 * @author Franck van Breugel
 */
public class Experiment {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		try {
			// Generates the labelled Markov chain from the input file
			LabelledMarkovChain chain = new LabelledMarkovChain(args[0]);
			long start = 0;
			long end = 0;
			
			// Experiments
			for(int i=0; i<55; i++) {		
				System.out.println("Itr: " + i);
				
				Explanation explanation = new Explanation(chain);
			
				// Records the time to obtain an optimal policy
				start = System.currentTimeMillis();
				explanation.optimalPolicy();
				end = System.currentTimeMillis();
				System.out.println("Optimal: " + (end - start)  + "ms");

				// Records the time to obtain an one maximal optimal policy
				start = System.currentTimeMillis();
				explanation.oneMaximize();
				end = System.currentTimeMillis();
				System.out.println("oneMaximal: " + (end - start)  + "ms");
	
				// Records the time to obtain a zero minimal one maximal optimal policy
				start = System.currentTimeMillis();
				explanation.zeroMinimize();
				end = System.currentTimeMillis();
				System.out.println("zeroMinimal: " + (end - start)  + "ms");
				
			}
						
			
		} catch (IllegalArgumentException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		

	}

}
