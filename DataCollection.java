/**
 * Used to collect data from both WNB and NB using the Edited LWL provided in the Github Folder
 */
package Research_LWL_WNB.LWL_WNB;

import java.util.Arrays;

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

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		Instances tmpInstances = null;

        //Finds data
        NaiveBayes model = new NaiveBayes();
        tmpInstances = (new DataSource("https://dl.dropboxusercontent.com/u/88175668/binary/heart-statlog/heart-statlog.arff")).getDataSet();
	
        double[] weights = new double[tmpInstances.numAttributes()];
        
        //The Naives bayes is edited to have a weight system added.
        //To Allow Normal Naive Bayes all weights must be 1
        Arrays.fill(weights, 1);
        model.setWeight(weights);

        //Sets the cutoff point in the data
        int cutoff = (int)(tmpInstances.numInstances() * .8);
        
        //Creates both the train and test data
        Instances train = new Instances(tmpInstances,0,cutoff);
        Instances test = new Instances(tmpInstances,cutoff,tmpInstances.numInstances() -cutoff);
        
        
        //Sets the class
        train.setClass(train.attribute(train.numAttributes()-1));
        test.setClass(test.attribute(train.numAttributes()-1));
        
        //Builds the model
        model.buildClassifier(train);
        
        for(int i =0; i < test.numInstances(); i ++)
        {
        	
        }
	}
	
	/**
	 * Finds the LWL Weights
	 * @return
	 * @throws Exception 
	 */
	public static double[] getWeights(Instances train,Instance test) throws Exception
	{        
        LWL LWLModel = new LWL();
        LWLModel.buildClassifier(train);
        double[] LWLWeights = LWLModel.getWeightsForDistributionForInstance(test);
        
		return LWLWeights;
		
	}

}
