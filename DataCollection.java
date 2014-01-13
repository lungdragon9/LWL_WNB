/**
 * Used to collect data from both WNB and NB using the Edited LWL provided in the Github Folder
 */
package Research_LWL_WNB.LWL_WNB;

import java.util.Arrays;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.ThresholdCurve;
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
	public static int[][][] costMatrixAW = new int[2][2][5];
	public static int[][][] costMatrixBW = new int[2][2][5];
	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		Instances tmpInstances = null;

        //Finds data
        NaiveBayes model = new NaiveBayes();
        tmpInstances = (new DataSource("C:\\WorkSpace\\binary\\colic\\colic.arff")).getDataSet();
	
        
        for(int k =0; k < 5; k++)
        {
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
        double[] newWeightInstandes = new double[train.numInstances()];
        Instances weightedInstances = new Instances(train);
        for(int i =0; i < test.numInstances(); i ++)
        {
        	//For Basic model without any weightsAttributes
        	//Builds the Naive Bayes Model
        	model.buildClassifier(train);
            model.setWeight(weightsAttributes);
            costMatrixNWModifier(model.distributionForInstance(test.instance(i)),test.instance(i));
            
            
            //Gets the LWL Weights
            newWeightInstandes =  getweightsInstances(train,test.instance(i)); 
            //Sets the LWL Weights
            for(int a =0; a < train.numInstances(); a++)
            {
            	weightedInstances.instance(a).setWeight(newWeightInstandes[a]);
            }
            model.buildClassifier(weightedInstances);
            model.setWeight(weightsAttributes);
            costMatrixIWModifier(model.distributionForInstance(test.instance(i)),test.instance(i));
            
            for(int a =0; a < 5; a ++)
            {
                Arrays.fill(newWeightAttributes, 1);
            	model.buildClassifier(train);
            	newWeightAttributes = MCMC(train,newWeightAttributes,.05,.00001);
            	model.setWeight(newWeightAttributes);
            	costMatrixAWModifier(model.distributionForInstance(test.instance(i)),test.instance(i),a);
            	
            	model.buildClassifier(weightedInstances);
            	model.setWeight(newWeightAttributes);
            	costMatrixBWModifier(model.distributionForInstance(test.instance(i)),test.instance(i),a);
            }
        	
        }
        costMatricOutputNW();
        costMatricOutputIW();
        costMatricOutputAW();
        costMatricOutputBW();
        }
	}
	
	/**
	 * Finds the LWL weightsAttributes
	 * @return
	 * @throws Exception 
	 */
	public static double[] getweightsInstances(Instances train,Instance test) throws Exception
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
	
	public static void costMatrixAWModifier(double[] predicted, Instance actual, int run)
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
				costMatrixAW[0][0][run] ++;
			}
			if(classValue == 1)
			{
				costMatrixAW[0][1][run] ++;
			}
		}
		if(predictedValue == 1)
		{
			if(classValue == 0)
			{
				costMatrixAW[1][0][run] ++;
			}
			if(classValue == 1)
			{
				costMatrixAW[1][1][run] ++;
			}
		}
	}
	
	public static void costMatrixBWModifier(double[] predicted, Instance actual, int run)
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
				costMatrixBW[0][0][run] ++;
			}
			if(classValue == 1)
			{
				costMatrixBW[0][1][run] ++;
			}
		}
		if(predictedValue == 1)
		{
			if(classValue == 0)
			{
				costMatrixBW[1][0][run] ++;
			}
			if(classValue == 1)
			{
				costMatrixBW[1][1][run] ++;
			}
		}
	}
	public static void costMatricOutputNW()
	{
		double a = costMatrixNW[0][0], b = costMatrixNW[0][1], c = costMatrixNW[1][0], d = costMatrixNW[1][1];
		
		System.out.println("No Weights");
		System.out.println("Accuracy : " + (a+d)/(a+b+c+d));
		System.out.println("Recall : " + (a)/(a+b));
		System.out.println("Precision : " + (a)/(a+c));
		System.out.println("F-Measure : " + (2*a)/(2*a+b+c));
	}
	public static void costMatricOutputIW()
	{
		double a = costMatrixIW[0][0], b = costMatrixIW[0][1], c = costMatrixIW[1][0], d = costMatrixIW[1][1];
		
		System.out.println("\n\n Instance Weights");
		System.out.println("Accuracy : " + (a+d)/(a+b+c+d));
		System.out.println("Recall : " + (a)/(a+b));
		System.out.println("Precision : " + (a)/(a+c));
		System.out.println("F-Measure : " + (2*a)/(2*a+b+c));
	}

	public static void costMatricOutputAW()
	{
		double a = ((costMatrixAW[0][0][0]+ costMatrixAW[0][0][1] + costMatrixAW[0][0][2] + costMatrixAW[0][0][3] + costMatrixAW[0][0][4])/5), 
			   b = ((costMatrixAW[0][1][0]+ costMatrixAW[0][1][1] + costMatrixAW[0][1][2] + costMatrixAW[0][1][3] + costMatrixAW[0][1][4])/5), 
			   c = ((costMatrixAW[1][0][0]+ costMatrixAW[1][0][1] + costMatrixAW[1][0][2] + costMatrixAW[1][0][3] + costMatrixAW[1][0][4])/5), 
			   d = ((costMatrixAW[1][1][0]+ costMatrixAW[1][1][1] + costMatrixAW[1][1][2] + costMatrixAW[1][1][3] + costMatrixAW[1][1][4])/5);
		
		System.out.println("\n\n Attribute Weights");
		System.out.println("Accuracy : " + (a+d)/(a+b+c+d));
		System.out.println("Recall : " + (a)/(a+b));
		System.out.println("Precision : " + (a)/(a+c));
		System.out.println("F-Measure : " + (2*a)/(2*a+b+c));
	}
	
	public static void costMatricOutputBW()
	{
		double a = ((costMatrixBW[0][0][0]+ costMatrixBW[0][0][1] + costMatrixBW[0][0][2] + costMatrixBW[0][0][3] + costMatrixBW[0][0][4])/5), 
			   b = ((costMatrixBW[0][1][0]+ costMatrixBW[0][1][1] + costMatrixBW[0][1][2] + costMatrixBW[0][1][3] + costMatrixBW[0][1][4])/5), 
			   c = ((costMatrixBW[1][0][0]+ costMatrixBW[1][0][1] + costMatrixBW[1][0][2] + costMatrixBW[1][0][3] + costMatrixBW[1][0][4])/5), 
			   d = ((costMatrixBW[1][1][0]+ costMatrixBW[1][1][1] + costMatrixBW[1][1][2] + costMatrixBW[1][1][3] + costMatrixBW[1][1][4])/5);
		
		System.out.println("\n\n Both Weights");
		System.out.println("Accuracy : " + (a+d)/(a+b+c+d));
		System.out.println("Recall : " + (a)/(a+b));
		System.out.println("Precision : " + (a)/(a+c));
		System.out.println("F-Measure : " + (2*a)/(2*a+b+c));
	}
	 /**
     * Finds the weights using Markov Chain Monte Carlo
     * @param train : training set
     * @param weights[] : Used for combines methods
     * @return : returns the weights
     * @throws Exception 
     */
    public static double[] MCMC(Instances train,double weights[], double deltaW, double eAUC) throws Exception
    {
    	ThresholdCurve curvefinder = new ThresholdCurve();
    	//Can be switched to just an double without the array
    	double AUCPost = 0;
    	double AUC=0;
    	double AUCDelta = 0;
    	NaiveBayes model = new NaiveBayes();
    	
    	double[] weights_local = new double[weights.length];
    	
    	//Splits the training set into two smaller sets
    	int cutoff = (int)(train.numInstances() * .2);
        
        //Creates both the train and test data
        Instances train_MCMC = new Instances(train,0,cutoff);
        //Helps to validate the changes to the weights
        Instances validate_MCMC = new Instances(train,cutoff,train.numInstances() -cutoff);
    	
		model.buildClassifier(train_MCMC);        
		
		Evaluation eval = new Evaluation(train_MCMC);
    	
    	int[] weights_direction = new int[weights.length];
    	
    	//Find which equation to use
    	alterWeightDirection(weights_direction);
    	
    	while(true) 
    	{
    		model.setWeight(weights);
    		eval.evaluateModel(model, validate_MCMC);
    		AUC = ThresholdCurve.getROCArea(curvefinder.getCurve(eval.predictions()));
    		
    		//weightDisplay(weights);
    		
    		for(int i = 0; i < weights.length; i++)
    		{
    			weights_local[i] = weights[i];
    			
    			switch (weights_direction[i])
    			{
	    			case 0:
	    				weights[i] += deltaW;
	    				break;
	    			case 1:
	    				weights[i] -= deltaW;
	    				break;
	    			case 2:
	    				break;
    			}
    			
    			//Used to test to see if negative weights affect the quality of Percent Correct
//    			if (weights[i] < 0)
//    			{
//    				weights[i] = 0;
//    			}
    		}
    		
    		model.setWeight(weights);
    		eval.evaluateModel(model, validate_MCMC);
    		AUCPost = ThresholdCurve.getROCArea(curvefinder.getCurve(eval.predictions()));
    		
    		AUCDelta = AUCPost - AUC;
    		
    		if (AUCDelta < 0)
    		{
    			alterWeightDirection(weights_direction);
    			
    			//Sets the weights back to make sure that no loss in quality was recorded
    			for(int i = 0; i < weights.length; i++)
        		{
    				weights[i] = weights_local[i];
        		}
    		}
    		else if (AUCDelta < eAUC)
    		{
    			break;
    		}
    	}
    	
		return weights;
    }
    
    
    public static void weightDisplay(double[] weights)
    {
    	int i =0;
    	for(i =0; i < weights.length; i ++)
    	{
    		System.out.print(weights[i] + " ");
    	}
    	System.out.println();
    }
    
    /**
     * Used by MCMC to randomize the direction of equations
     * @param weights_direction A list of ints to show weather + or - or nothin
     */
    public static void alterWeightDirection(int[] weights_direction) 
    {
    	Random r = new Random();
    	r.setSeed(System.currentTimeMillis());
    	int Low = 0;
    	int High = 3;
    	
    	for(int i =0; i < weights_direction.length; i++)
		{
			weights_direction[i] = r.nextInt(High-Low) + Low;
		}
    }

}
