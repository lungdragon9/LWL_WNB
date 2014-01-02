/**
 * Used to collect data from both WNB and NB using the Edited LWL provided in the Github Folder
 */
package Research_LWL_WNB.LWL_WNB;

import java.util.Arrays;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.LWL;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * @author stamps
 *
 */
public class DataCollection {

	
	public static int[][] costMatrixNW = new int[2][2];
	public static int[][] costMatrixIW = new int[2][2];
	public static int[][] costMatrixBW = new int[2][2];
	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		Instances tmpInstances = null;

        //Finds data
        NaiveBayes model = new NaiveBayes();
        tmpInstances = (new DataSource("C:\\WorkSpace\\binary\\mushroom\\data.arff")).getDataSet();
	
        //Randomize the dataset
        tmpInstances.randomize(tmpInstances.getRandomNumberGenerator(System.currentTimeMillis()));
        
        double[] weightsAttributes = new double[tmpInstances.numAttributes()];
        
        //The Naives bayes is edited to have a weight system added.
        //To Allow Normal Naive Bayes all weightsAttributes must be 1
        Arrays.fill(weightsAttributes, 1);
        
        model.setWeight(weightsAttributes);

        //Sets the cutoff point in the data
        int cutoff = (int)(tmpInstances.numInstances() * .8);
        
        //Creates both the train and test data
        Instances train = new Instances(tmpInstances,0,cutoff);
        Instances test = new Instances(tmpInstances,cutoff,tmpInstances.numInstances() -cutoff);
        

        double[] weightsAttributesInstances = new double[train.numInstances()];
        Arrays.fill(weightsAttributesInstances, 1);
        
        //Sets the class
        train.setClass(train.attribute(train.numAttributes()-1));
        test.setClass(test.attribute(train.numAttributes()-1));
        
        //Builds the model
        model.buildClassifier(train);
        
        
        double[] newWeightAttributes = new double[tmpInstances.numAttributes()];
        for(int i =0; i < test.numInstances(); i ++)
        {
        	//For Basic model without any weightsAttributes
        	//Builds the Naive Bayes Model
            model.setWeight(weightsAttributes);
            costMatrixNWModifier(model.distributionForInstance(test.instance(i)),test.instance(i));
            
            
        	
        }
        costMatricOutputNW();
	}
	
	/**
	 * Finds the LWL weightsAttributes
	 * @return
	 * @throws Exception 
	 */
	public static double[] getweightsAttributes(Instances train,Instance test) throws Exception
	{        
        LWL LWLModel = new LWL();
        LWLModel.buildClassifier(train);
        double[] LWLweights = LWLModel.getWeightsForDistributionForInstance(test);
        
		return LWLweights;
		
	}
	
	public static void costMatrixNWModifier(double[] predicted, Instance actual)
	{
		double classValue = actual.classValue();
		double predictedValue;
		if(predicted[0] > predicted[1])
		{
			predictedValue = 0;
		}
		else
		{
			predictedValue = 1;
		}
		
		if(predictedValue == 0)
		{
			if(classValue == 0)
			{
				costMatrixNW[0][0] ++;
			}
			if(classValue == 1)
			{
				costMatrixNW[0][1] ++;
			}
		}
		if(predictedValue == 1)
		{
			if(classValue == 0)
			{
				costMatrixNW[1][0] ++;
			}
			if(classValue == 1)
			{
				costMatrixNW[1][1] ++;
			}
		}
	}
	
	public static void costMatrixIWModifier(double[] predicted, Instance actual)
	{
		double classValue = actual.classValue();
		double predictedValue;
		if(predicted[0] > predicted[1])
		{
			predictedValue = 0;
		}
		else
		{
			predictedValue = 1;
		}
		
		if(predictedValue == 0)
		{
			if(classValue == 0)
			{
				costMatrixIW[0][0] ++;
			}
			if(classValue == 1)
			{
				costMatrixIW[0][1] ++;
			}
		}
		if(predictedValue == 1)
		{
			if(classValue == 0)
			{
				costMatrixIW[1][0] ++;
			}
			if(classValue == 1)
			{
				costMatrixIW[1][1] ++;
			}
		}
	}
	public static void costMatricOutputNW()
	{
		double a = costMatrixNW[0][0], b = costMatrixNW[0][1], c = costMatrixNW[1][0], d = costMatrixNW[1][1];
		
		System.out.println("Accuracy : " + (a+d)/(a+b+c+d));
		System.out.println("Recall : " + (a)/(a+b));
		System.out.println("Precision : " + (a)/(a+c));
		System.out.println("F-Measure : " + (2*a)/(2*a+b+c));
	}

}
