/**
 * @version 1.00  May 2001
 * @author Nevin L. Zhang
 */

package hnb;

import java.io.*;
import java.util.*;
import hlcm.*;
import hnb.ScoreCalculator;
import hnb.ParameterLearner;


/**
   *  This class implements algorithms for determining cardinalities
   * of hidden variables. So far, only one algorithm is available. The
   * Algorithm starts with all variables being binary and does hill climbing.
   *
   * The EM stop threshold for determining
   * cardinalities of hidden variables  is set at 0.01. This
   * can be changed by configuring the ParameterLeaner, whih can
   * be got using getParameterLearner.
   *<p>
  *  To configure a CardinalityLeaner, one need to
  * <ul>
  * <li>setEMStopThreshold4Size,   setOutputDirectory, and setProgressInfoDisplayLevel
  * <li> getParameterLearner and getScoreCalculator and configure them properly.
  * </ul>*
   *
   */
public class CardinalityLearner
{
	public CardinalityLearner()
	{
		paraLearner.setStopThreshold(0.01);
	}


	/**
	  * Decompose model structure and call hill Climb on each component
	  */
	public HNB decomposeHillClimb(HNB modelStructure, LabelledDataTable table)
	{
		HNB[] components = modelStructure.decompose();

		for (int i=0; i< components.length; i++)
		{
			HNB model_i = components[i];

			LabelledDataTable table_i =
			    (LabelledDataTable)table.project( model_i.getObservedVariables() );

			components[i] = hillClimb(model_i, table_i);
		}

		return HNB.merge( components);

	}

	/**
	  * Determine domains sizes for the hidden variables by hill climbing
	  * from modelStructure. Parameters in modelStructures are ignored.
	  * Input is not altered.
	  */
	public HNB hillClimb(HNB modelStructure, LabelledDataTable table)
	{


		// start with the case where all hidden variables
		// are binary.
		HNB model =  (HNB) modelStructure.copy();
		setAllHiddenVarsBinary(model);

		// this is so that the progress info make sense
		model.setID( modelStructure.getID() );

		return hillClimbContinue( model, table );
	}

	/**
	  * Continue hill climbing from model. Current cardinalities and
	  * parameters of model are respected. But parameters are reestimated
	  * just so that the settings are the same for different cardinality
	  * settings.
	  */
	public HNB hillClimbContinue( HNB model, LabelledDataTable table)
	{
		HNB bestModel = paraLearner.EM(model, table);
		double bestScore = scoreCal.computeScore(bestModel, table);


		int step = 0;

		// Hill climbing
		boolean done = false;
		do {
			step++;


			HNB[] modelList = generateNewModels(bestModel);
			int N = modelList.length;

			double nextBestScore = - HLCM.INFNTY;
			HNB nextBestModel = null;
			boolean firstModel = true;
			for (int i=0; i<N; i++)
			{

				modelList[i] = paraLearner.EM(modelList[i], table );
				double score = scoreCal.computeScore(modelList[i], table);


				// when L2 is used, the score is very close to zero.
				// this first condition ensures the program works even in
				// that situation.
				if (firstModel || score > nextBestScore)
				{
					nextBestScore = score;
					nextBestModel = modelList[i];
				}
				firstModel = false;
			}

			// this happens if there are no next models.
			if (nextBestModel == null) break;


			// determine whether to continue
			if ( nextBestScore > bestScore)
			{
				bestScore = nextBestScore;
				bestModel = nextBestModel;
				done = false;
			}
			else
				done = true;

		} while (!done);


		return bestModel;
	}



	/**
	  * Set the size of all hidden variables of an HLCM to 2
	  */
	public void setAllHiddenVarsBinary(HNB hlcm)
	{
			setAllHiddenVarsBinaryRecursive( hlcm.getRoot() );
	}

	private void setAllHiddenVarsBinaryRecursive( Node node)
	{
		// reset all prob distributions
		if ( !node.hasParent() )
			node.setPriorProb( null );
		else
			node.setCondProb( null );


		if ( node.hasChildren() ) // work only on hidden varables
		{
			if ( node.hasParent() )      // root is observed
				node.getVar().setSize(2);  // where work is done.

			Iterator iter = node.getChildren().iterator();
			while (iter.hasNext() )
			{
				Node child = (Node) iter.next();
				setAllHiddenVarsBinaryRecursive( child );
			}
		}
	}


	/**
	  * Generate new models by increasing the sizes of hidden
	  * variables, one at a time. This way we get an array of new models.
	  */
	private HNB[] generateNewModels(HNB model)
	{
		Node.List hiddenNodes = model.hiddenNodes();
		int N = hiddenNodes.size();

		HNB[] newModels = new HNB[N];

		for (int i=0; i<N; i++)
		{
			newModels[i] = (HNB) model.copy();
			Node nodeInModel = newModels[i].getNode4Var( hiddenNodes.getNode(i).getVar() );

			// reset prob distributions for all
			// nodes whose probs are affected.
			if ( !nodeInModel.hasParent() )
				nodeInModel.setPriorProb( null );
			else
				nodeInModel.setCondProb( null );

			Iterator iter = nodeInModel.getChildren().iterator();
			while (iter.hasNext() )
			{
				Node tmpNode = (Node) iter.next();
				tmpNode.setCondProb( null );
			}


			// increase size
			Variable varInModel = nodeInModel.getVar();
			varInModel.setSize( varInModel.getSize() +1 );

		}
		return newModels;
	}


	/**
	  * Get ParameterLearner for this CardinalityLearner. (One might want
	  * to modify the scoring metric.)
	  */
	public ParameterLearner getParameterLearner()
	{
		return paraLearner;
	}

	/**
	  * Get ScoreCalculator for this CardinalityLearner. (One might want
	  * to modify the scoring metric.)
	  */
	public ScoreCalculator getScoreCalculator()
	{
		return scoreCal;
	}
	/**
	  * Set score calculator to use
	  */
	public void setScoreCalculator( ScoreCalculator cal )
	{
		scoreCal = cal;
	}
	/**
	  *  Show settings of this cardinality learner:
	  *  the settings of parameter learner and score caluators
	  */
	public void showSettings(PrintWriter out)
	{
		out.println("Settings of cardinality Learner:");
		out.print("  Of cardinality Learner:");
		paraLearner.showSettings( out );
		out.print("  Of Cardinality Learner: ");
		scoreCal.showSettings( out );

		out.flush();
	}


	/**
	  * Progress information display is at the momemt
	  * muted because I have not figured out what information
	  * might be useful for user.
	  * <p>
	  * Set level of details at which progress info will be displayed
	  * to System.out. Alternatives include:
	  * <ul>
	  * <li>0 (the default):  don't display anything
	  * <li> 1: Show score of best model fond at each step and write
	  *			the model to file outputDirectory + "/c.m." + step + ".txt"
	  * <li> 2:  In addition to the above, print scores of alternative
	  *          models examined at each step and write them to files
	  *		outputDirectory + "/c.m." + step + "." + modelID + ".txt"
	  * </ul>
	  */
	public void setProgressInfoDisplayLevel( int level )
	{
		progressInfoDisplayLevel = level;
	}

	public int getProgressInfoDisplayLevel( )
	{
		return progressInfoDisplayLevel;
	}

	/**
	  * Set the directory into which results are to be written.
	  * The default is "models".  Make sure the directory exist.
	  * The program does not create directory.
	  */
	public void setOutputDirectory( String dirName )
	{
		outputDirectory = dirName;
	}

	public String getOutputDirectory( )
	{
		return outputDirectory;
	}

	private String outputDirectory = "models";

	private ParameterLearner paraLearner  = new ParameterLearner();
	private ScoreCalculator scoreCal  = new ScoreCalculator();
	private int progressInfoDisplayLevel = 0;

}