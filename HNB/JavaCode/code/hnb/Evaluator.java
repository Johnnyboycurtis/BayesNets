

/**
 * @version 1.00  January 2002
 * @author Nevin L. Zhang
 */

package hnb;

import java.io.*;
import java.util.*;
import hnb.*;
import hnb.StructureLearner;
import cern.jet.stat.*;
import hlcm.*;

/**
  * Class for evaluating individual LSB models
  * and evaluating our HNB-based supervised learning method.
  */
public class Evaluator
{
	/**
	  * Classification error of model on test data.
	  */
	public double classificationErrorRate( HNB model, LabelledDataTable testData )
	{
		double rate =  (double) classificationError(model, testData)/testData.getTotalWeight();

		return rate;
	}

	/**
	  * Classification error
	  */
	int classificationError( HNB model, LabelledDataTable testData )
	{
		int error =0;
		for (int i=0; i<testData.getSize(); i++)
		{
			DataTable.Record record = testData.getRecord(i);

			int label = model.classify(record, testData);
			int label1 = testData.classLabelOfRecord( record );

			if (label != label1 )
				error += record.getWeight();
		}
		return error;
	}

	/**
	  * Classification accuracy of model on test data.
	  * It is 1.0 - error rate.
	  */
	public double classificationAccuracy( HNB model, LabelledDataTable testData )
	{
		return 1.0 - classificationErrorRate( model,testData );
	}



	/**
	  * Half of the length of confidence interval. Estimated
	  * using +-1.96 * SD = 1.96 * sqrt (accuracy (1- accuracy) /N )
	  */
	public double confidenceInterval(double accuracy,
												int N)
	{
		return 1.96*Math.sqrt( accuracy * (1.0 - accuracy) / (double )N);
	}


	/**
	  * Cross validation. Returns classification accuracy.  One can
	  * call confidenceInterval to get the confidence interval.
	  * <p>
	  * The method will first partition the data file into "fold" roughly
	  * equal portions and portions are written to subfile dataFileName+"."+i.
	  * Before the partitioning, the method would check whether
	  * the subfiles exists. If they do, then no need to partition.
	  * <p>
	  * Cross validation is conducted in the usual way.  Intermediate
	  * results will be written outputDirectory+"/"learningMethod+"/".
	  * Make sure the directory exists.
	  * It is advised that the user redirct the messages print to
	  * standard io to the same directory.
	  * <p>
	  * The model for each fold is written as intermediate results.
	  * The score filed of the model is actual the classification accuracy
	  * of the model on the testing data for the fold.
      *
	  * @param learningMethod Name of learning method.  "NaiveBayes", "HNB", or
	  *				"HNBHeuristic".
	  * 			"NaiveBayes" is assumed if the name is not spelled correctly.
	  * @param dataFileName Name of data file.
	  * @param fold  Number of folds
	  * @param outputDirectory Directory for output.
	  *
	  */
	public double crossValidation(String learningMethod, String dataFileName,
							int fold, String outputDirectory)
	{
		System.out.println("Evaluating " + learningMethod);

		// test to see if dataFileName+"."+0 exists
		File tmpFile = new File( dataFileName +"."+0);
		if ( !tmpFile.exists() )
		{
			LabelledDataTable table = (LabelledDataTable) LabelledDataTable.readData( dataFileName );

			for (int i=0; i<fold-1; i++)
			{
				DataTable[] subtables = table.split( 1.0 /(double) (fold-i));
				subtables[0].printToFile( dataFileName +"."+i);

				table = (LabelledDataTable)subtables[1];
			}
			table.printToFile( dataFileName +"."+ (fold-1)); // the last portion
		}

		int totalError = 0;
		double totalWeight =0.0;    // total number of testing records


		for (int i=0; i<fold; i++)
		{
			LabelledDataTable training = null;
			LabelledDataTable testing = null;

			// prepare data for this fold.
			for (int j=0; j<fold; j++)
			{
				LabelledDataTable tmp= (LabelledDataTable)
							LabelledDataTable.readData(dataFileName+"."+j);
				if (i==j)
					testing = tmp;
				else if (training == null)
					training = tmp;
				else
					training  = training.merge(tmp);
			}


			// learning Model
			HNB model = null;

			if (learningMethod.equals("HNB") )
			{
				model = HNB.learnHNB( training,
										outputDirectory + "/" + learningMethod,
										2);
			}
			else if (learningMethod.equals("HNBHeuristic") )
				model = HeuristicLearner.learnHNB( training );
			else
				model = HNB.learnNBModel( training );


			// testing
			//Don't forget synchronize variable IDs!!!
			model.synchronizeVarIDs(testing.getHeader() );
			double error = classificationError( model, testing);
			model.setScore(1.0- error / testing.getTotalWeight() );


			model.printToFile(outputDirectory + "/" + learningMethod
										   +"/FOLD." + i + ".model.txt");

		    totalError +=error;

			totalWeight += testing.getTotalWeight();
		}

		double accuracy = (totalWeight - totalError )/ totalWeight;

		return accuracy;
	}


}