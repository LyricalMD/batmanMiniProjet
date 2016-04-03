
import java.io.FileReader;
import java.io.PrintWriter;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.rules.ZeroR;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instances;

/**
 * Getting Started - a Minimal Working Example
 * 
 * Summary of the data
 * @author MicroBES
 *
 */
public class Statistics {

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
		
    	// Summary in string
    	//Num = numeric Nom = Nominal = Catégoriel
    	System.out.println("TRAINING DATA");
    	System.out.println(trainData.toSummaryString());
    	System.out.println("VALID DATA");
		System.out.println(validData.toSummaryString());
		System.out.println("TEST DATA");
		System.out.println(testData.toSummaryString());
		System.out.println("Success!");

    }
 	
 }
