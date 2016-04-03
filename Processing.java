import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instances;

/**
 * Getting Started - a Minimal Working Example
 * 
 * Obtain the best classifier by comparing results from cross-validation
 *
 */
public class Processing {

	public static void main(String[] args) throws Exception {
    	
    	String dataPath = "/home/boris/batman/data2/";
    	String challengeName = "batman3"; 
    	
    	System.out.println("Getting Started with " + challengeName + "!");
        
    	// Path for the 3 datasets
    	String trainPath = dataPath + challengeName + "/train.arff";
    	String validPath = dataPath + challengeName + "/valid.arff";
    	String testPath = dataPath + challengeName + "/test.arff";
   	
    	// Create instances
    	Instances trainData = new Instances(new FileReader(trainPath));
    	Instances validData = new Instances(new FileReader(validPath));
    	Instances testData = new Instances(new FileReader(testPath));
		

    	
    	// Set the attribute to predict (the last one) in each dataset
    	int ind = trainData.numAttributes() - 1;
    	trainData.setClassIndex(ind);
    	validData.setClassIndex(ind);
    	testData.setClassIndex(ind);

    	
    	// Extract the attribute to predict 
    	// in order to convert predictions index to prediction label later
    	Attribute Resolution = trainData.attribute(ind);
 
    
    	Random rand = new Random(1);
		int folds = 10;	
		
		// Define an Evaluation object to predict
		Evaluation validEval = new Evaluation(trainData);
		Evaluation testEval = new Evaluation(trainData);
		Evaluation eval = new Evaluation(trainData);

		//	Define a list which contains the classifiers to compare
		ArrayList<Classifier> list = new ArrayList<Classifier>();
		list.add(new ZeroR());
		list.add(new OneR()); 
		list.add(new NaiveBayes()); 
    	list.add(new J48()); 
    	list.add(new DecisionStump());
    	list.add(new JRip());
    	for(int i = 0; i < list.size(); i++){
    		
    	//Define each model and train it
    		list.get(i).buildClassifier(trainData);
    		
    	//FastVector to store the results for the valid dataset
    		validEval.evaluateModel(list.get(i), validData);
    		FastVector validPred = validEval.predictions();
    		
    	// The same for the test dataset
	    	testEval.evaluateModel(list.get(i), testData);
	    	FastVector testPred = testEval.predictions();
    	
	    // Define a PrintWriter to save predicted value in files
    	// NOTE : submitted files on Codalab must have the name "valid.predict" and "test.predict" 	
	    	PrintWriter pw;
	    	
	    	
	    	// Save the predicted values from the valid dataset
	    	pw = new PrintWriter(i + "ref_valid.predict" , "UTF-8");
	    	for (int j = 0; j < validPred.size(); j++) {
				double val = ((NominalPrediction) validPred.elementAt(i)).predicted();
				pw.print(Resolution.value((int) val) + "\n");
			}
	    	pw.close();
	    	
	    	// Save the predicted values from the test dataset
	    	pw = new PrintWriter( i+  "ref_test.predict", "UTF-8");
	    	for (int k = 0; k < testPred.size(); k++) {
				double val = ((NominalPrediction) testPred.elementAt(i)).predicted();
				pw.print(Resolution.value((int) val) + "\n");
			}
	    	pw.close();
	    	
	    	//Cross-validation
    	eval.crossValidateModel(list.get(i), trainData, folds, rand);
		double scoreWithCv = eval.areaUnderROC(1);
		System.out.println("Estimated Accuracy for "+ list.get(i).getClass()+ ": " +Double.toString(eval.pctCorrect()));
		System.out.println("Estimated Accuracy for "+ list.get(i).getClass()+ " with CV: " +Double.toString(scoreWithCv));
		
    	
    	}
 
    	System.out.println("Success!");
   }
	
}
		    

	
