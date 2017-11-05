/**
 * @version 1.00  May 2001
 * @author Nevin L. Zhang
 */

package hnb;


 import java.io.*;
 import java.util.*;
import hlcm.*;



 /**
  * This class is for calculating score of an LSB model. The parameters
  * in the model must be MLE. User can choose
  * among different scoring metrics by using setScoringMetric( String metricName).
  * Currently, the following alternatives are availables: AIC,  BIC, Cheeseman-Stutz,
  * Logarithmic score (on validation data), goodness-of-fit via L-sqaured,
  * an approximation of Monte Carlo. Currently only BIC is implemented and it is
  * the default.
  * <p>
  * Note that there is also a static method for computing loglikelihood.
  * <p>
  * To configure a ScoreCalculator, one needs to setScoringMetric.
  * When the scoringMetric is CS, one also needs to setEffectiveSampleSize.
  */

public class ScoreCalculator
{

	public ScoreCalculator() {};

	/**
	  * Choose scoring metric for comparing models. Alternatives available:
	  *  "LS", "AIC", "BIC", "L2", "MC",  and "CS", where "LS" stands for
	  *   "Logarithmic score", "L2" stands for L-Squared statistic,
	  *  "MC" stands for Monte Carlo,
	  * "CS" stands for "Cheeseman-Stutz". "CS" is
	  * the default. The logarithmic score is on validation data. As such,
	  * one should make sure to pass the correct data table to the method
	  * computeScore.
	  *
	  */
	public void setScoringMetric( String metricName )
	{
		scoringMetric = metricName;
	}

	/**
	  * Get the current scoring metric.
	  */
	public String getScoringMetric()
	{
		return scoringMetric;
	}


	/**
	  * Log Probability of a data set: log-likelihood of model
	  * and store it in the model.
	  * Make sure that  all observed
	  * variables of model must appear in tableHeader.
	  * The table header can contain other variables and the order
	  * does not matter.
	  *
	  */
	public static double computeLoglikelihood(HNB model,  LabelledDataTable table )
	{

		double result = 0.0;

		int N = table.getSize();
		for (int i=0; i<N; i++)
		{
			DataTable.Record datum = table.getRecord( i );
			result += Math.log( model.probOfDatum( datum, table)) *
				   datum.getWeight();  // for data aggregation
		}

		return result;
	}

	/**
	  * Computes score of a model in light of a "table" of data.
	  * Use setScoringMetric(String) to select
	  * an scoring metric. The score is stored in the score field of model
	  */
	public double computeScore( HNB model, LabelledDataTable table )
	{
		double score = -HLCM.INFNTY;
		if ( scoringMetric.equals("BIC")  )
		{
			score = BICScore( model, table );
			model.setScore( score );
		}
		else
		{
			throw new HNBException("Score metric " + scoringMetric +
							" not available." );
		}

		return score;
	}

	// Computes BIC score: log P(D|theta^, model) - d logN/2
	private double BICScore( HNB model, LabelledDataTable table)
	{
		double logL =  computeLoglikelihood(model, table );
		model.setLoglikelihood( logL );

		return logL -  model.dimension() * Math.log(table.getTotalWeight())/2.0;

	}

	/**
	  *  Show settings of this score calculator,
	  */
	public void showSettings(PrintWriter out)
	{
		out.println("Settings of ScoreCalculator:");
		out.println("   Scoring Metric: "	+ scoringMetric);

		out.flush();
	}

	private String scoringMetric = "BIC";

}



