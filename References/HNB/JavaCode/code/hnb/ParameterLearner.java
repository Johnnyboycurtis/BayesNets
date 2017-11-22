
/**
 * @version 1.00  January 2002
 * @author Nevin L. Zhang
 */

package hnb;

import java.io.*;
import java.util.*;
import cern.jet.stat.*; // a statistics package
import hlcm.*;
import hnb.ScoreCalculator;




/**
  * This class implements algorithms for learning parameters.
  * <p>
  * To configure a ParameterLeander, one needs to
  *   setLocalMaximaEscapeMethod,  setNumInitIterations,   setNumOfStartPoints,
  *    setOutputDirectory, setProgressInfoDisplayLevel,
  *      setStopThreshold, and  setMaxEMSteps.
  */

public class ParameterLearner
{
	public ParameterLearner(){};



	/**
	  * This is a more efficient version of EM. It takes advantage of the fact
	  * that an LSB model can be decomposed into several components according
	  * children of the root and EM can be run independently in each component.
	  */
	public HNB decomposeEM( HNB modelStructure, LabelledDataTable table)
	{
		HNB[] components = modelStructure.decompose();

		for (int i=0; i< components.length; i++)
		{
			HNB model_i = components[i];
			LabelledDataTable table_i =
			    (LabelledDataTable)table.project( model_i.getObservedVariables() );

			components[i] = EM(model_i, table_i);

		}

		return HNB.merge( components);
	}


	/**
	  * Learn parameters for a model from data using EM.
	  * EM terminate if either the change in likelihood
	  * becomes sufficiently small or the total
	  * number of iterations exceeds MAX_EM_STEPS.
	  *
	  * @param modelStructure A LSB model structure. Numerical information
	  *                       of structure is disregarded.
	  * @param table A data table whose header constains all
	  *		observed variables in model.
	  *			   <b>  If not, program terminates</b>.
	  *
	  * @return An LSB model, whose parameters are estimated
	  *  		by EM. Loglikelihood of model is also computed.
	  */

	public HNB EM( HNB modelStructure, LabelledDataTable table )
	{


		if  ( getLocalMaximaEscapeMethod().equals("ChickeringHeckerman" ))
			return EM_ChickeringHeckerman(modelStructure, table);
		else
		{
			throw new HNBException("Local maxima escape method " +
						getLocalMaximaEscapeMethod() + "  not available");
		}
	}



	/**
	  * Learn parameters for a model from data using EM.
	  * Local maxima escapsed using method by (Chickering & Heckerman 1997).
	  *
	  * @param modelStructure An LSB model structure. Numerical information
	  *                       is structure is disregarded.
	  * @param table A data table whose header constains all
	  *		observed variables in model.
	  *			   <b>  If nt, program terminates</b>.
	  * @param numOfStarts  Number of random starting points. Must be power of
	  *  		2.
	  * @return An LSB model, whose parameters are estimated
	  *  		by EM.  The imput model stucture is altered.
	  */
	private HNB EM_ChickeringHeckerman(
				HNB modelStructure, LabelledDataTable table)
	{
		/* ------- multiple start to avoid local maximum -----*/
		int N = getNumOfStartPoints() ;
		HNB nextModel = null;
		double nextL = - HLCM.INFNTY;
		double prevL = - HLCM.INFNTY;

		HNB currentModel	= null;
		double currentL 	= - HLCM.INFNTY;

		// Create N initial models from model structure
		HNB[] modelList = new HNB[N];
		for (int i=0; i<N; i++)
		{
			nextModel=  (HNB) modelStructure.copy();
			nextModel.randomParameterize();

			// run several EM steps on each model
			for (int k=0; k<getNumInitIterations(); k++)
			{
				nextModel = oneStepEM( nextModel, table );
			}
			nextL = ScoreCalculator.computeLoglikelihood(nextModel, table );
			nextModel.setLoglikelihood( nextL );

			modelList[i] =  nextModel ;
		}

		int numIterations = 1; // number of iterations before comparing
							   // and removing models from consideration
		while ( N > 1)
		{
			// number of models
			int numModels = modelList.length;

			// perform EM on each model numIterations times
			for (int j=0; j<numModels; j++)
			{
				currentModel = modelList[j];

				for (int i=0; i<numIterations; i++)
				{
					nextModel = oneStepEM( currentModel, table );

					// compute loglikelihood at the last iteration
					if (i == numIterations -1)
					{
						nextL = ScoreCalculator.computeLoglikelihood(nextModel, table );
						nextModel.setLoglikelihood( nextL );
					}

					currentModel = nextModel;
				}
				// replace old model with new model
				modelList[j] = nextModel;
			}

			// sort models according to loglikelihood in ascending order
			Arrays.sort( modelList );

			// Remove first half of the models
			HNB[] modelList1 = new HNB[ numModels/2 ];
			for (int k=0; k< numModels/2; k++)
					modelList1[k] = modelList[ k + numModels/2 ];

			modelList = modelList1;

			numIterations = numIterations*2;
			N = N/2;
		}

		/* -------now only one model left -----*/
		nextModel= modelList[0];
		currentL = nextModel.getLoglikelihoodOfPrevModel();

		// number of past iterations
		do
		{
			currentModel = nextModel;
			prevL = currentL;

			nextModel = oneStepEM( currentModel, table );
			currentL = nextModel.getLoglikelihoodOfPrevModel();
			if ( getProgressInfoDisplayLevel()>0)
			{
				System.out.print("Iteration: "
									+ nextModel.getNumEMSteps()
									+ " loglike: " );
				System.out.println( currentL );
				nextModel.printToFile( getOutputDirectory()
							+ "/M.EMCurrent.txt");
			}

		}  while ( (  currentL - prevL ) >= getStopThreshold()
					&& nextModel.getNumEMSteps() <
					hlcm.ParameterLearner.MAX_EM_STEPS);

		return nextModel;
	}



	/**
	  * One step of EM.
	  */
	public HNB oneStepEM( HNB model, LabelledDataTable table )
	{
		// compute sufficient statistics
		HNB suffStats = M_Step( model, table );

		// normalize to get an HLC model
		suffStats.normalize();

		suffStats.setNumEMSteps( model.getNumEMSteps() + 1 );

		return suffStats;
	}

	/**
	  * The M_step of EM algorithms. Computes sufficient statistics
	  * and stored the results in an LSB model.
	  * In the E_step, one simply normalizes this model.
	  */
	public  HNB M_Step( HNB model, LabelledDataTable table)
	{
		double loglikelihoodOfModel = 0.0;
		HNB suffStats = (HNB) model.copy();
		suffStats.resetProbs();

		// another copy of model to be used as clique tree
		HNB cliqueTree = (HNB) model.copy();

		int N = table.getSize();
		for (int i=0; i<N; i++)
		{
			DataTable.Record datum = table.getRecord(i);

			// after these lines, cliqueTree  and model are the same
			cliqueTree.resetProbs();
			cliqueTree.addProbs( model );


			// collection: propagate toward root
			messageCollection( cliqueTree.getRoot(), table, datum );

			// distribution: propagat away from root.
			// compute probability of data under current model
			// as by-product
			double probOfDatum =
			             messageDistribution( cliqueTree.getRoot(), table, datum );

			loglikelihoodOfModel +=  datum.getWeight() *Math.log( probOfDatum );

			// normalize clique marginals
			cliqueTree.normalizeCliqueMarginals();


			// multiply all probability with datum.getWeight()
			cliqueTree.multiplyConstantProbs( datum.getWeight() );

			// add probs of clique tree to those of suffStats
			suffStats.addProbs( cliqueTree);
		}


		suffStats.setLoglikelihoodOfPrevModel( loglikelihoodOfModel );
		return suffStats;
	}


	/*** NOTE for the following two functions:
		Inference is done via propagation. The clique tree consists
		of a clique for each node and its parents. This clique is represented
		by the node itself. There is a special clique for the root, which functions
		as a separater. This clique is represented by the root itself.

	**/

	// Collection: propagate messages toward the clique represented by "node" from
	// the "descendants cliques".
	// Messages from children cliques  are collected and stored at the node.
	// The messages are also combined and the combined message stored also at the node.
	// All messages are function of node.getVar() only.
	// Original probs associated with the node (potential associated with
	//  the node-clique) is not altered.
	private void messageCollection( Node node,
						LabelledDataTable table, DataTable.Record datum )
	{
		// marginal condition
		if ( !node.hasChildren() )
		// one message from "child", i.e. the evidence
		{
			int index = table.getHeader().indexOf( node.getVar() );
			if ( index < 0 )
				throw new HNBException("lsbm.ParameterLearner.messageCollection:"
						+ " observed variable # "
						+   node.getVar().getID()
						+ " not in table " );

			int state = datum.getCell(index);
			Function1V evidence = null;
			// the evidence is the identity function is value
			// for the variable is missing. Otherwise it is a
			// characteristic function.
			if ( state == DataTable.MISSING)
				 evidence = Function1V.identity( node.getVar().copy() );
			else
				evidence =	Function1V.characteristic( node.getVar().copy(), state );

			Function1V[] messages = new Function1V[1];
			messages[0] = evidence;

			// store array of messages from children
			node.setMsgArrayChildren( messages );
			// store combine messages from children
			node.setCombinedMsgChildren( evidence.copy() );
		}
		else {

			Node.List children = node.getChildren();
			int n = children.size();
			Function1V[] messages= new Function1V[ n ];

			for (int i=0; i<n; i++)
			{
				Node child = children.getNode(i);

				// recursive call
				messageCollection( child, table, datum );

				// compute message from this child:
				Function2V tmpFun =  child.getCondProb().copy();
				tmpFun.multiply( child.getCombinedMsgChildren() );
				messages[i] = tmpFun.marginalizeOut( child.getVar() );

			}
			// store array of messages
			node.setMsgArrayChildren( messages );

			// compute combination of messages and store it
			Function1V combinedMsg = Function1V.identity( node.getVar() );
			for (int j=0; j<messages.length; j++)
						combinedMsg.multiply( messages[j]);
			node.setCombinedMsgChildren( combinedMsg );

		}

	}

	// Distribution: combine all messages sent to the node-clique (together
	// with potential associated with the clique) and
	// propagate message from the node-clique toward its children
	// cliques. If node if root, return the sum of function values
	// which is the probability of the data case
	private double messageDistribution( Node node, LabelledDataTable table,
										DataTable.Record datum)
	{
		double probOfDatum = 0.0;

		Variable var = node.getVar();
		Function1V[] msgChildren = node.getMsgArrayChildren();

		// combine all incoming messages and the potential associated
		// with this node-clique
		Function1V combineMsgChildren =	 node.getCombinedMsgChildren();

		if ( !node.hasParent() ) // if node is the root
		{
			// obsorb evidence
			Function1V prior = node.getPriorProb();
			Function1V evidence =
						Function1V.characteristic( node.getVar(),
									table.classLabelOfRecord( datum ));
			prior.multiply( evidence );

			prior.multiply( combineMsgChildren);
			probOfDatum = node.getPriorProb().getTotal();
		}
		else
		{
			node.getCondProb().multiply( combineMsgChildren);
			node.getCondProb().multiply( node.getMsgParent() );
		}

		// send message to children cliques. The message is a function
		// of node.getVar()!!!
		if ( node.hasChildren() )
		{
			Node.List children = node.getChildren();
			int n = children.size();
			for (int i=0; i<n; i++)
			{
				Node child = children.getNode( i );

				// the root clique is a special clique.
				// sending messages to its children cliques does
				// not require marginalization
				if ( !node.hasParent() )
				{
					// message to child[i] = combined message /
					//  					msg from child[i]
					Function1V msg2Child = node.getPriorProb().copy();

					msg2Child.divide( msgChildren[i] );

					child.setMsgParent( msg2Child );
				}
				else
				{
					// message to child[i] = \sum_node combined message /
					//  					msg from child[i]
					Function1V msg2Child =  node.getCondProb().marginalizeOut(
												  node.getParent().getVar() );

					msg2Child.divide( msgChildren[i] );


					child.setMsgParent( msg2Child );
				}

				// recursive call
				messageDistribution( child, table, datum );
			}
		}

		return probOfDatum;

	}


	/**
	  * Choose a method for escaping local maxima in EM.
	  * One alternative, namely "ChickeringHeckerman", is
	  * currently available.
	  */
	public  void setLocalMaximaEscapeMethod( String methodName )
	{
		localMaximaEscapeMethod = methodName;
	}

	public  String getLocalMaximaEscapeMethod(  )
	{
		return localMaximaEscapeMethod;
	}

	/**
	  *  Choose number of start points for escaping local maxima in EM.
	  *  The default is 32.
	  */
	 public void setNumOfStartPoints( int n )
	 {
		 numOfStartPoints = n;
	 }

	public int getNumOfStartPoints( )
	 {
		 return numOfStartPoints;
	 }
	 /**
	  *  Choose number of initial EM iterations to run one each
	  *  initial model. The default is 10.
	  */
	 public void setNumInitIterations( int n )
	 {
		 numInitIterations = n;
	 }
	 public int getNumInitIterations(  )
	 {
		 return numInitIterations;
	 }
	/**
	  *  Choose threshold for EM to stop. The default is 0.000001
	  */
	public void setStopThreshold( double epsilon )
	{
		stopThreshold = epsilon;
	}
	public double getStopThreshold(  )
	{
		return stopThreshold;
	}
	/**
	  *  Choose the maximum number of EM iterations that
	  * are allowed on a model.
	  */
	public void setMaxEMSteps( int n )
	{
		MAX_EM_STEPS = n;
	}

	public int getMaxEMSteps( )
	{
		return MAX_EM_STEPS ;
	}

	/**
	  *  Show settings of this parameter learner
	  */
	public void showSettings(PrintWriter out)
	{
		out.println("Settings of parameter Learner:");
		out.println("   Method for escaping local maxima: "
							+ localMaximaEscapeMethod);
		out.println("   Number of starting points for hilling climbing: "
							+ numOfStartPoints);
		out.println("   Stopping threshold: " +
							+ stopThreshold);
		out.println("   Max EM Steps: " +
							+ MAX_EM_STEPS);
		out.flush();
	}

	/**
	  * Set level of details at which progress info will be displayed
	  * to System.out. Alternatives include:
	  * <ul>
	  * <li>0 (the default):  don't display anything
	  * <li> 1: show loglikelihood after each iteration and
	  *       the current model written to outputDirectory+"/M.EMCurrent.txt".
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

	/**
	  * Maximum number of EM iteration allowed for any
	  * model. The default is 200.
	  */
	public static int MAX_EM_STEPS = 200;
	private String outputDirectory = "models";


	private int progressInfoDisplayLevel = 0;
	private String localMaximaEscapeMethod = "ChickeringHeckerman";
	private int numOfStartPoints = 32;
	private int numInitIterations = 0;
	private  double stopThreshold = 0.000001;
}