
import mic.Comparateur;
import java.io.FileReader;
import java.io.PrintWriter;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.meta.Stacking;
import weka.classifiers.meta.Vote;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
public class CombineModels {
	
	public static void main(String[] args) throws Exception {
		
		String dataPath = "/home/boris/batman/data2/";
    	String challengeName = ""; 
    	
    	System.out.println("We're comparing " + challengeName +"'s data !");
        
    	// Path for the 3 datasets
    	String trainPath = dataPath + challengeName + "/train.arff";
    	String validPath = dataPath + challengeName + "/valid.arff";
    	String testPath = dataPath + challengeName + "/test.arff";
   	
    	// Create instances
    	Instances trainData = new Instances(new FileReader(trainPath));
    	Instances validData = new Instances(new FileReader(validPath));
    	Instances testData = new Instances(new FileReader(testPath));
		
		//get instances object 
		//set class index .. as the last attribute
    	int ind = trainData.numAttributes() - 1;
    	trainData.setClassIndex(ind);
    	validData.setClassIndex(ind);
    	testData.setClassIndex(ind);
    	
    	//
    	Comparateur newC = new Comparateur();
    	newC.getScores();
    	newC.SelectScores();
    	newC.SortClassifiers();
    	
		Classifier[] classifiers = {				
					newC.t[1].nom,
					newC.t[2].nom,
					newC.t[3].nom
			};
		/*for (int i =0; i < 3; i++){
			classifiers[i].setOptions(newC.t[i].options);
		}*/
			
			Vote voter = new Vote();
			voter.setClassifiers(classifiers);//needs one or more classifiers
			voter.buildClassifier(trainData);
			Stacking stacker = new Stacking();
			stacker.setMetaClassifier(new NaiveBayes());//needs one meta-model
			
			stacker.setClassifiers(classifiers);//needs one or more models
			stacker.buildClassifier(trainData);
			
			// Extract the attribute to predict 
	    	// in order to convert predictions index to prediction label later
	    	Attribute Resolution = trainData.attribute(ind);
	 
	 
	 
	    	
	    	// Define an Evaluation object to predict and FastVector to store the results for the valid dataset
	    	Evaluation validEval = new Evaluation(trainData);
	    	validEval.evaluateModel(stacker, validData);
	    	FastVector validPred = validEval.predictions();
	    	
	    	
	    	// The same for the test dataset
	    	Evaluation testEval = new Evaluation(trainData);
	    	testEval.evaluateModel(stacker, testData);
	    	FastVector testPrediction = testEval.predictions();
	    	
	    	
	    	// Define a PrintWriter to save predicted value in files
	    	// NOTE : submitted files on Codalab must have the name "valid.predict" and "test.predict" 	
	    	PrintWriter pw;
	    	
	    	
	    	// Save the predicted values from the valid dataset
	    	pw = new PrintWriter("ref_valid.predict", "UTF-8");
	    	for (int i = 0; i < validPred.size(); i++) {
				double val = ((NominalPrediction) validPred.elementAt(i)).predicted();
				pw.print(Resolution.value((int) val) + "\n");
			}
	    	pw.close();
	    	
	    	
	    	// Save the predicted values from the test dataset
	    	pw = new PrintWriter("ref_test.predict", "UTF-8");
	    	for (int i = 0; i < testPrediction.size(); i++) {
				double val =  ((NominalPrediction) testPrediction.elementAt(i)).predicted();
				pw.print(Resolution.value((int) val) + "\n");
			}
	    	pw.close();
			//System.out.println("Estimated Accuracy for stacking "+ ": " +Double.toString(validEval.pctCorrect()));
	    	System.out.println("Success!");

	   }
		}
	
