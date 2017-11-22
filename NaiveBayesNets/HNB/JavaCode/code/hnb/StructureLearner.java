
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
import hnb.ParameterLearner;
import hnb.CardinalityLearner;


/**
  *  Class for learning HNB model structure from data.
  *  Itermediate results will be written to a
  *  outputDirectory, which is by default "models"
  * and can be changed using the method setOutputDirectory.
  *  Make sure the directory exists.
  * <p>
  * To configure a StructureLeaner, one needs to
  * <ul>
  * <li> setStructureLearningMethod, setModelVars,  setOutputDirectory, and setProgressInfoDisplayLevel
  * <li> getCardLearner and getScoreCalculator and configure them properly.
  * </ul>
  */
public class StructureLearner
{

	/**
	  * Don't forget to set the data table when using
	  * this default constructor.
	  */
	public StructureLearner()
	{	}




	/**
	  *  Find the best model structure among a list of HNB structures.
	  *  Parameters and score of the resulting model set.
	  * <p>
	  * In all the models given, the observed variables must be a subset
	  * of variables in the header of the table.  The method does the following:
	  *  <ul>
	  *  <li> Call a CardinalityLearner is  to learn the
	  *  cardinalities of hiddens variabls in each structure.
	  *  (In this process, parameters for the best model with
	  *   the structure are also computed.)
	  *  <li> Compare the best models and pick the best one.
	  *  </ul>
	  */
	public HNB findBestModel( ArrayList modelList, LabelledDataTable table)
	{
		double bestScore = - HLCM.INFNTY;
		HNB bestModel = null;
		boolean  firstModel = true;

		Iterator iter = modelList.iterator();
		while ( iter.hasNext() )
		{
			HNB model = (HNB) iter.next();

			model = cardLearner.decomposeHillClimb(model, table );

			double score =scoreCalculator.computeScore(model, table );

			if (progressInfoDisplayLevel >=2 )
				System.out.println("   SL: Score of model " + model.getID()
								   + " = " + score );
			if (progressInfoDisplayLevel >=3 )
				model.printToFile(outputDirectory + "/m.SL."+  algoStep +"."+ model.getID() + ".txt");

			// the first condition is to avoid some numerical
			// problems (when the score is very small when L2 is used )
			if ( firstModel || bestScore < score )
			{
				bestScore = score;
				bestModel = model;
			}
			firstModel = false;
		}

		return bestModel;
	}






	/**
	  * HillClimb starting from initModel based on data "table".
	  * initModel is not altered otherwise.
	  */
	private HNB hillClimb( HNB initModel, LabelledDataTable table )
	{
		System.out.println("SL: starting hill climb");

		double bestScore = scoreCalculator.computeScore( initModel, table );
		HNB bestModel = initModel;

		boolean done = false;
		do {
			ArrayList neighborModels = neighborModels(  bestModel );

			// if no neighbors
			if ( neighborModels.size() == 0)
				return bestModel;

			HNB nextBestModel = findBestModel( neighborModels, table );
			double nextBestScore = nextBestModel.getScore();

			if (progressInfoDisplayLevel >=1 )
			{
				System.out.println("SL: Step "+ algoStep + " done. "
									+ " bestScore: " + nextBestScore
									+ " bestModel: " + nextBestModel.getID() );
				nextBestModel.printToFile(outputDirectory + "/M.SL."+ algoStep + ".best.txt");
			}

			if ( nextBestScore > bestScore )
			{
				bestScore = nextBestScore;
				bestModel = nextBestModel;
			}
			else
				done = true;

			algoStep++;
		} while (!done);

		if (progressInfoDisplayLevel >=1 )
		{
			System.out.println("SL: hill climb done. BestScore = " + bestScore);
			bestModel.printToFile(outputDirectory + "/M.SL.best.txt");

		}
		return bestModel;
	}



	/**
	  * Create init model and HillClimb based on data "table"
	  * @para table: datatable to use instead of default datatable of learner.
	  */
	public HNB hillClimb(LabelledDataTable table )
	{
		HNB initModel  =learnNBModel ( table.getHeader(), table );

		return hillClimb( initModel, table );
	}



	/**
	  * Learn a naive Bayes model (a special HNB model) on modelVars
	  * from the data table provided. The last member of
	  * modelVars is the class variable.
	  * The variables are copied.
	  */
	 HNB learnNBModel( Variable.List modelVars, LabelledDataTable table)
	{
		Variable rootVar = modelVars.getVar( modelVars.size() - 1 );

		Node root = new Node( rootVar.copy() );
		HNB model = new HNB( root);

		for (int i=0; i<modelVars.size() -1; i++)
		{
			Variable var = modelVars.getVar(i);

			Node child = new Node( var.copy() );
			child.makeParent( root );
		}

		paraLearner.setNumOfStartPoints( 1 );
		model = paraLearner.decomposeEM(model, table);
		double score = scoreCalculator.computeScore( model, table);
		model.setScore( score );
		if (progressInfoDisplayLevel >=1 )
		{
			System.out.println("SL: Done constructing Naive Bayes model. "
								+ "Score: " + score);
			model.printToFile(outputDirectory + "/M.SL.0.txt");
			algoStep++;
		}
		return model;
	}

	/**
	  * ArrayList of all models that can be constructed from curHLCM
	  * by applying two operators: parentAlteration and latentNodeIntroduction.
	  */
	public ArrayList neighborModels( HNB curModel )
	{
		ArrayList modelList = new ArrayList();

		newModelsViaParentAlteration( curModel, modelList );

		newModelsViaNewParentIntroduction( curModel, modelList );

		newModelsViaNodeDeletion( curModel, modelList );

		return modelList;
	}


	/**
	  * Construct all models by altering parents of nodes in curModel and
	  * add the new models to modelList.
	  * <p>
	  * For each nonleaf node n and a child c1 of n,
	  * we (1)if n is not the root,  create a new model by changing
	  * the parent of c1 to the parent of n;
	  * and (2) create a new model by change the parent of c1 to
	  * another child of n that is a nonleaf node.
	  * While doing so, we make sure that we don't create chains of
	  * two singly connected latent nodes.
	  * </ul>
	  */
	private void newModelsViaParentAlteration( HNB curModel, ArrayList modelList)
	{
		// get list of nodes with children
		Node.List  nonLeafNodes  = curModel.hiddenNodes();
		nonLeafNodes.add(0, curModel.getRoot().copy() ); // add root

		// consider the nonleaf nodes one by one
		Iterator iter = nonLeafNodes.iterator();
		while ( iter.hasNext() )
		{
			Node node = (Node) iter.next();
			Node nodeInModel = curModel.getNode4Var(node.getVar() );

			// get children node
			Node.List children = nodeInModel.getChildren();
			int N = children.size() ;

			if (N<2) continue;

			// consider each combination of children
			for (int i=0; i<N; i++)
			{
				Node child1 = children.getNode( i );

				// whether we can change the parent of child1 to
				// its grand parent, i.e. the parent of node.
				boolean canChangeToGrandParent = false;

				if ( nodeInModel.hasParent() ) // node has parent
				{
					// if node has more than 2 children,
					// the parent alteration does not make node singly connected.
					if ( N>2)
						canChangeToGrandParent = true;
					else  // node has only two children
					{
						Node theOtherChild = null;

						if (i==0)
							theOtherChild = children.getNode(1);
						else
							theOtherChild = children.getNode(0);

						// if the other child is not singly connected,
						// (either it has not children or has more than1)
						// the parent alteration can go ahead
						if ( !theOtherChild.hasChildren() ||
							 theOtherChild.getChildren().size() >=2 )
							canChangeToGrandParent = true;
					}
				}


				if ( canChangeToGrandParent )
				{

					Node parent = nodeInModel.getParent();
					HNB newModel = (HNB) curModel.copy();

					// the nodes in the new model
					Node nodeInNewModel = newModel.getNode4Var( node.getVar() );
					Node child1InNewModel = newModel.getNode4Var( child1.getVar() );
					Node parentInNewModel = newModel.getNode4Var( parent.getVar() );


					// change the parent of child1 from latenNode to child 2
					child1InNewModel.unmakeParent( nodeInNewModel );
					child1InNewModel.makeParent( parentInNewModel );

					modelList.add( newModel );
				}


				for (int j=0; j<N; j++)
				{
					Node child2 = children.getNode(j );

					//  whether we can make child1 a child of child2
					boolean canChangeToChild2 = false;

					// basic conditions
					if (  (i != j) && (child2.hasChildren()) )
					{
						// also need to make sure that we do
						// not create chain of two singly connected nodes
						if ( N >2 )
							canChangeToChild2  = true;
						else
						{
							// in this case, parent alteration will make node singly
							// connected. we need to make sure that the parent of
							// node, if exists,  is not singly connected.
							if ( !nodeInModel.hasParent() ||
								nodeInModel.getParent().getChildren().size() >2)
								canChangeToChild2  = true;
						}
					}

					if ( canChangeToChild2)
					{
						HNB newModel = (HNB) curModel.copy();

						// the nodes in the new model
						Node nodeInNewModel = newModel.getNode4Var( node.getVar() );
						Node child1InNewModel = newModel.getNode4Var( child1.getVar() );
						Node child2InNewModel = newModel.getNode4Var( child2.getVar() );


						// change the parent of child1 from latenNode to child 2
						child1InNewModel.unmakeParent( nodeInNewModel );
						child1InNewModel.makeParent( child2InNewModel );


						modelList.add( newModel );

					}
				}
			}

		}


	}



	/**
	  * Construct all models by introducing new parent to  nodes in curModel and
	  * add the new models to modelList.
	  * <p>
	  * For each non-leaf node that has two or more children, we
	  * create a new model for each pair of  its children
	  * by introducing a new parent for them as long as this does not
	  * create a chain of two singly connected nodes.
	  *
	  *
	  * </ul>
	  */
	private void newModelsViaNewParentIntroduction( HNB curModel, ArrayList modelList)
	{
		// get list of nodes with children
		Node.List  nonLeafNodes  = curModel.hiddenNodes();
		nonLeafNodes.add(0, curModel.getRoot().copy() ); // add root

		// consider the nonleaf nodes one by one
		Iterator iter = nonLeafNodes.iterator();
		while ( iter.hasNext() )
		{
			Node node = (Node) iter.next(); // this is a copy of
										    // a node in model

			Node nodeInModel = curModel.getNode4Var( node.getVar() );

			// get children node
			Node.List children = nodeInModel.getChildren();
			int N = children.size() ;

			if ( N<2) continue;

			// consider each combination of children
			for (int i=0; i<N; i++)
			{
				Node child1 = children.getNode( i );

				for (int j=i+1; j<N; j++)
				{
					Node child2 = children.getNode(j );

					// whether we can introduce a new parent for child1 and child2
					boolean canIntroduceNewParent = false;

					if (   N>2  // no singly connected node can be created inthis case.
						|| !nodeInModel.hasParent()  // nodeInModel is the root
						||  nodeInModel.getParent().getChildren().size() >2
							// parent of nodeInModel is not singly connected
					   )
						canIntroduceNewParent = true;


					//  introduce new parent for child1 and child2

					if (canIntroduceNewParent )
					{
						HNB newModel = (HNB) curModel.copy();

						// the nodes in the new model
						Node nodeInNewModel = newModel.getNode4Var( node.getVar() );
						Node child1InNewModel = newModel.getNode4Var( child1.getVar() );
						Node child2InNewModel = newModel.getNode4Var( child2.getVar() );


						// remove current links
						child1InNewModel.unmakeParent( nodeInNewModel );
						child2InNewModel.unmakeParent( nodeInNewModel );

						// new node
						Node newNode = new Node( new Variable(2) );

						// introduce new links
						child1InNewModel.makeParent( newNode );
						child2InNewModel.makeParent( newNode );

						newNode.makeParent( nodeInNewModel );

						modelList.add( newModel );
					}
				}
			}

		}

	}



	/**
	  * Construct all models by deleting latent nodes
	  * that has only one or two children
	  *
	  * </ul>
	  */
	private void newModelsViaNodeDeletion( HNB curModel, ArrayList modelList)
	{
		// get list of nodes with children
		Node.List  nonLeafNodes  = curModel.hiddenNodes();

		// consider the latent nodes one by one
		Iterator iter = nonLeafNodes.iterator();
		while ( iter.hasNext() )
		{
			Node node = (Node) iter.next(); // this is a copy of
										    // a node in model

			Node nodeInModel = curModel.getNode4Var( node.getVar() );

			// we cannot delete nodeInModel if it has more than
			// 2 children.
			if ( nodeInModel.getChildren().size() >2 )
				continue;

			// create a new model by deleting nodeInModel
			HNB newModel = (HNB) curModel.copy();

			// the nodes in the new model
			Node nodeInNewModel = newModel.getNode4Var( node.getVar() );

			Node newParent = nodeInNewModel.getParent();
			nodeInModel.unmakeParent( newParent );

			Node.List children = nodeInNewModel.getChildren();
			while (children.size() >0 )
			{
				Node child = children.getNode( 0);
				child.unmakeParent( nodeInNewModel );
				child.makeParent( newParent);
			}

			modelList.add( newModel );
		}

	}


	/**
	  *  Show settings of this structure learner:
	  *  the method of structure learning,
	  *   the settings of the parameter learner,
	  *  score caluators, and score calculator for this
	  *  structure learner. The paramter learner is only
	  *  for refine the parameter of the final model.
	  */
	public void showSettings(PrintWriter out)
	{
		out.println("Settings of structure Learner:");
		out.println("   Structure learning method:" + structureLearningMethod);
		out.print("  Of structure learner** ");
		cardLearner.showSettings(out );
		out.print("  Of structure learner**");
		scoreCalculator.showSettings( out );

		out.flush();
	}

	/**
	  * Get the cardinality learner used by this structure learner.
	  * One can then modify the settings of the cardinality learner.
	  */
	public CardinalityLearner getCardLearner()
	{
		return cardLearner;
	}

	/**
	  * Get the score calculator used by this structure learner.
	  * One can then modify the settings of the calculator.
	  */
	public ScoreCalculator getScoreCalculator()
	{
			return scoreCalculator;
	}

	/**
	  * Set method of structure learning. Available alternatives include
	  * "IncrementalModelLearning", "HillClimbing", and "HillClimbingWithLogScore".
	  *"IncrementalModelLearning" is the default.
	  * For the "HillClimbingWithLogScore" option, you must do the hillClimbLogScoreSetup.
	  */
	public void setStructureLearningMethod( String methodName )
	{
		structureLearningMethod = methodName;
	}
	/**
	  * Get method of structure learning method.
	  */
	public String getStructureLearningMethod()
	{
		return structureLearningMethod;
	}


	/**
	  * Tell the StructureLearner the data table based on which learning
	  * to take place. The default model variables is set
	  */
	public void setTable( LabelledDataTable table )
	{
		masterTable = table;
		setModelVars( table.getHeader() );
	}


	/**
	  * Tell the StructureLearner the observed variables for the
	  * model to be constructed.  The default is those on the
	  * header of the data table for the structureLearner.
	  */
	public void setModelVars( Variable.List modelVars )
	{
		this.modelVars = modelVars;
	}

	/**
	  *Set level of details at which progress info will be displayed
	  * to System.out. Alternatives include:
	  * <ul>
	  * <li>0 (the default):  don't display anything
	  * <li> 1: signalify the completion of each major step
	  *       	      and print the best model to file
	  *				   outputDirectory + "/M." + algoStep +" .txt".
	  *
	  * <li> 2:  In addition to the above, print scores of alternative models.
	  * <li> 3:  In addition to the above, print alternative models to files
	  * 		  outputDirectory + "/m."+ algoStep + modelID + ".txt"
	  * </ul>
	  */
	public void setProgressInfoDisplayLevel( int level )
	{
		progressInfoDisplayLevel = level;
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

	private String outputDirectory = "models";
	private String structureLearningMethod = "IncrementalModelLearning";

	private int progressInfoDisplayLevel = 0;

	private Variable.List modelVars = null;
	private LabelledDataTable masterTable = null;

	private ScoreCalculator scoreCalculator = new ScoreCalculator();
	private CardinalityLearner cardLearner = new CardinalityLearner();
	private ParameterLearner paraLearner = new ParameterLearner();

	// keep track the number of steps in algorithm
	private int algoStep = 0;
}