

/*
Inspired by :
https://github.com/nsadawi/WEKA-API/blob/master/src/Evaluate.java
*/

import weka.core.Instances;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;
import weka.classifiers.rules.JRip;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Classifier;
import weka.classifiers.rules.OneR;

/*Une classe utiliser comme structure pour pouvoir rasambler dans une meme variable un Classifier
 * le score du Classifier avec CV, le score sans CV, et les options de Classifier
 */
class ClassifierStruct{
	Classifier nom;
	double scoreSansCV;
	double scoreAvecCv;
	String[] options;
	public  void affichage(){
		Class cls = nom.getClass();
		System.out.println("Class: "+cls.getName());
		System.out.println("Score sans CV: "+scoreSansCV);
		System.out.println("Score avec CV : "+scoreAvecCv);
    	System.out.println("Options: "+Arrays.toString(options));
		System.out.println();

	}
}

/*Une classe utiliser comme structure pour pouvoir rasambler dans une meme variable un Classifier
 * avec ses options et avec le meilleure score, soit le score de CV soit le score sans CV
 */
class BestScores{
	Classifier nom;
	double score;
	boolean CV;
	String[] options ;
	
	public  void affichage(){
		Class cls = nom.getClass();
		System.out.println("Class: "+cls.getName());
		System.out.println("Score: "+score);
		System.out.println("CV : "+CV);
    	System.out.println("Options: "+Arrays.toString(options));
		System.out.println();

	}
}

public class Comparateur{
	public DataSource trainPath;
	public static Instances trainData;
	public ArrayList<ClassifierStruct> cTab = new ArrayList<ClassifierStruct>();
	public ArrayList<BestScores> sTab = new ArrayList<BestScores>();
	//public ArrayList<BestScores> sorted = new ArrayList<BestScores>();
	public BestScores[] t = new BestScores[5];
	//public BestScores[] top3 = new BestScores[3];

	
	/*Methode qui va essayer plusiers classifiers pour voir le score avec CV et sans CV de chaque classifier
	 * et va mettre  dans le tableau cTab le resultat de chaque classifier pour qu'on puisse apres choisir les meilleurs classifiers
	 */
	public void getScores() throws Exception{
		DataSource trainPath = new DataSource("/home/boris/batman/data2/train.arff");
		Instances trainData = trainPath.getDataSet();
		trainData.setClassIndex(trainData.numAttributes()-1);
		Evaluation eval = new Evaluation(trainData);
		Random rand = new Random(1);
		int folds = 10;		
		Classifier top;
		double bestScore = 0.0;
		boolean noCV = false;
		
		ArrayList<Classifier> list = new ArrayList<Classifier>();
		String[][] options;
		options = new String[5][5];

		list.add(new NaiveBayes());
		String [] option0 = new String[1];
		option0[0]="-D";
		list.add(new JRip());
    	String[] option1 = new String[3];
    	option1[0]="-F";	option1[1]="2"; 	option1[2]="-E";
		list.add(new J48());
    	String[] option2=new String[4];
    	option2[0]="-C";	option2[1] ="0.3";	option2[2] ="-M";	option2[3] ="1";
		list.add(new OneR());
		String [] option3 = new String[2];
		option3[0]="-B";	option3[1]= "10";
		list.add(new RandomForest());
		String [] option4 = new String[4]; 
		option4[0]="-I";	option4[1]="5";		option4[2]="-depth";	option4[3]="10";
		options[0] = option0;
		options[1] = option1;
		options[2] = option2;
		options[3] = option3;
		options[4] = option4;
		//System.out.println(Arrays.deepToString(options));

		top = list.get(0);
		
		ClassifierStruct tempC = new ClassifierStruct();
		for(int i = 0; i < list.size(); i++){

			tempC.nom = list.get(i);

			list.get(i).setOptions(options[i]);
			list.get(i).buildClassifier(trainData);
			eval.evaluateModel(list.get(i), trainData);
			tempC.options = list.get(i).getOptions();
			//System.out.println(Arrays.toString(tempC.options));

			tempC.scoreSansCV = (double)eval.areaUnderROC(1);
			if (bestScore < (double)eval.areaUnderROC(1)){
				top = list.get(i);
				bestScore = eval.areaUnderROC(1);
				noCV = false;
			}	
			eval.crossValidateModel(list.get(i), trainData, folds, rand);
			tempC.scoreAvecCv = (double)eval.areaUnderROC(1);
			
			if (bestScore < (double)eval.areaUnderROC(1)){
				top = list.get(i);
				bestScore = eval.areaUnderROC(1);
				noCV = true;
			}
			cTab.add(tempC);
			
		    tempC = new ClassifierStruct(); 
		}
	}
	
	/*Methode qui va comparer les scores avec CV et sans CV de chaque comparateur
	 * et va mettre dans sTab le nom de classifier, le meilleure score, les options, 
	 * et dans la variable CV il va mettre true si le meilleure score a ete avec CV et false sinon
	 */
	public void SelectScores(){
		BestScores temp = new BestScores();
		for (int i = 0; i < cTab.size(); i++){
			temp.nom = cTab.get(i).nom;
			if(cTab.get(i).scoreAvecCv > cTab.get(i).scoreSansCV){
				temp.CV = true;
				temp.score = cTab.get(i).scoreAvecCv;
			}else{
				temp.CV = false;
				temp.score = cTab.get(i).scoreSansCV;
			}
			temp.options = cTab.get(i).options;
			sTab.add(temp);
			temp = new BestScores();
		}
		
	}
	/*Methode qui va trier les tableau avec les meilleurs score des classifiers
	 *pour qu'on pouisse apres prendre un ou plusieurs classifiers
	 */
	public void SortClassifiers(){
		
		BestScores temp = new BestScores();
		boolean change = true;
		for (int i = 0; i < this.sTab.size(); i++){
			t[i] = this.sTab.get(i);
		}
		
		while(change){
			change = false;
			for (int i = 0; i<t.length-1; i++){
				if(t[i].score < t[i+1].score){
					temp=t[i];
					t[i]=t[i+1];
					t[i+1]=temp;
					change = true;
				}
			}
		}
	}
	
	public static void main(String args[]) throws Exception{
		Comparateur test = new Comparateur();
		test.getScores();
		test.SelectScores();
		test.SortClassifiers();
		
		for (int i = 0; i < test.t.length; i++){
			test.t[i].affichage();
			System.out.println();
		}		
	}
	
}

