/**
 * @version 1.00  January 2002
 * @author Nevin L. Zhang
 */

package hnb;  // latent structure Bayesian models

import java.io.*;
import java.util.*;
import java.text.*;
import hlcm.*;

/**
  *  HNB (Hierarchical Naive Bayesian model) is the same as
  *  HLCM except that the variable at root is observed. This
  *  implies that HNB behaves in the same way as HLCM when
  *  standing alone.  When interacting with data, however,
  *  it behaves differently.
  */

  public class HNB extends HLCM
  {
	public HNB() { super(); }

	public HNB( Node node) { super( node); }


	 /**
	  * Find all observed variables in model by depth-first
	  * search except that  the root variable is placed at the end
	  * of the the list.
	  * The returned variables are copies.
	  */
	public  Variable.List getObservedVariables()
	{
		Variable.List varList = super.getObservedVariables();

		Variable var = getRoot().getVar();

		varList.add( var.copy() );

		return varList;
	}

    /**
	  * Collect (copies) of all hidden nodes into an array
	  * according to the depth-first-search order.
	  */
	public Node.List hiddenNodes()
	{
		Node.List nodeList = super.hiddenNodes();

		nodeList.remove(0); // remove the root

		return nodeList;
	}


	/**
	  * Probability of a datacase in this model.  Leaf nodes
	  * of model do not have to correspond to any variables  in tableHeader
	  * and variable in tableHeader do not have to correspond to
	  * leaf nodes. Missing values are allowed but class label
	  * cannot be missing.
	  *  However, the root variable must be the last variable on the
	  * table header.
	  */
	public double probOfDatum(DataTable.Record datum, LabelledDataTable table )
	{
		Node root = getRoot();
		int classLabel = table.classLabelOfRecord( datum );
		Function1V resultFun =
		     Function1V.characteristic( root.getVar().copy(),
		       			 classLabel );

		resultFun.setVal(classLabel, getRoot().getPriorProb().getVal( classLabel ) );


		Iterator iter = root.getChildren().iterator();
		while (iter.hasNext() )
		{
			Node child = (Node) iter.next();
			Function1V tmpFun = message( child, root, datum, table.getHeader());
			resultFun.multiply( tmpFun );
		}

		// now resultFun is prior of root times message from children
		return resultFun.getTotal();
	}

	/**
	  * Message from child to parent
	  */
	private Function1V message(Node child, Node parent,
					DataTable.Record datum, Variable.List tableHeader )
	{
		Function1V resultFun = null;

		// try to find index of child variable in tableHeader
		Variable childVar = child.getVar();
		int index = tableHeader.indexOf( childVar );

		// if childVar is in table header
		if ( index >=0 )
		{
			// if the value for child is missing in datum,
			// we simply return an identity function of the parent variable
			if ( datum.getCell(index) == DataTable.MISSING)
				return Function1V.identity( parent.getVar().copy() );

			resultFun =  child.getLikelihood4Parent( datum.getCell( index ) );
		}
		else // recursion
		{

			// if child is a leaf node and does not appear in tableHeader,
			// then its value is not available in datum. In this
			// case, we simply return an identity function of the parent variable
			if ( !child.hasChildren() )
				return Function1V.identity( parent.getVar().copy() );

			Function2V jointFun = child.getCondProb().copy();

			Node.List grandChildren = child.getChildren();
			Iterator iter = grandChildren.iterator();
			while (iter.hasNext() )
			{
				Node grandChild = (Node) iter.next();
				Function1V tmpFun = message( grandChild, child, datum, tableHeader);
				jointFun.multiply( tmpFun );
			}

			resultFun = jointFun.marginalizeOut( child.getVar() );
		}
		return resultFun;
	}


	/**
	  * Decompose this model according to the children of the root.
	  * There is one component in correspondence to each children.
	  * Subtree rooted at children of the root are not touched.
	  * The root is duplicated and each copy linked to one
	  * child of the root.  This model is destroyed.
	  */
	public HNB[] decompose()
	{
		Node root = getRoot();
		Node.List childrenOfRoot = root.getChildren();

		int N  = childrenOfRoot.size();
		HNB[] components = new HNB[N];

		for (int i=0; i<N; i++)
		{
			Node rootCopy = root.copy();
			components[i] = new HNB( rootCopy );
			childrenOfRoot.getNode(i).makeParent( rootCopy);
		}

		return components;

	}

	/**
	  * This is the reverse of docompose.  components is an array of
	  * LSB models. In each of the models, the root has only one child.
	  * Moreover, the root variables and its prior probs are the
	  * same in all the components. This method merge them into one
	  * model.
	  */
	public static HNB merge( HNB[] components )
	{
		int N = components.length;
		HNB result = components[0];
		Node root = result.getRoot();

		for (int i=1; i<N; i++)
		{
			Node child = components[i].getRoot().getChildren().getNode(0);
			child.makeParent( root);
		}

		return result;
	}

	/**
	  *  Deep copy
	  */

	  	/**
	  	  *  Deep copy.
	  	  */
	  	public HLCM copy()
	  	{
	  		Node newRoot = recursiveNodeCopy( getRoot() );

	  		HNB newModel = new HNB( newRoot);

	  		return newModel;
	  	}

	  	private Node recursiveNodeCopy( Node node)
	  	{
	  		Node newNode = node.copy();

	  		Iterator iter = node.getChildren().iterator();
	  		while (iter.hasNext() )
	  		{
	  			Node child = (Node) iter.next();
	  			Node newChild = recursiveNodeCopy( child );
	  			newChild.makeParent( newNode );
	  		}

	  		return newNode;
	  	}



	/**
	  *  Print content to a PrintWriter stream
	  */
	public void show(PrintWriter out)
	{
		out.println("// HNB model in the BIF format");
		out.println("// Produced by the HNB package");
		out.println("");
		out.println("network \"" + getName() +"\" {}"  );
		out.println("");

		showNodeRecursively( getRoot(), out );
		showProbRecursively( getRoot(), out );

		out.println("//Loglikelihood:  " + getLoglikelihood());
		out.println("//Score        :  " + getScore() );
		out.println("//LoglikelihoodOfPreviousModel:  " +
					getLoglikelihoodOfPrevModel());
		out.flush();
	}

	// Show variable information. Later we will add position of
	// node with each variable.
	private void showNodeRecursively( Node node, PrintWriter out)
	{
	  	node.showVar(out);
	  	out.println();
		Node.List children = node.getChildren();
		for (int i=0; i < children.size(); i++)
			showNodeRecursively( children.getNode(i), out);

	}

	private void showProbRecursively( Node node, PrintWriter out)
		{
			node.showProb(out);
			out.println();
			Node.List children = node.getChildren();
			for (int i=0; i < children.size(); i++)
				showProbRecursively( children.getNode(i), out);

	}


	/**
	  * Learn an HNB from LabelledDataTable.
	  * Intermediate results written to "outputDirectory". It's
	  * the caller's responsibility to make sure that
	  * the directory exists.  What get written depends
	  * on the progressInfoDisplayLevel, which also
	  * controls what info are displayed to standout.
	  *  Alternatives include:
	  * <ul>
	  * <li>0 (the default):  don't display anything.
	  * <li> 1: signalify the completion of each major step
	  *       	      and print the best model to file
	  				   outputDirectory + "/M." + algoStep +" .txt".
	  *
	  * <li> 2:  In addition to the above, print scores of alternative models.
	  * <li> 3:  In addition to the above, print alternative models to files
	  * 		  outputDirectory + "/m."+ algoStep + modelID + ".txt"
	  * </ul>
	  */
	public static HNB learnHNB( LabelledDataTable table,
								   String outputDirectory,
								   int progressInfoDisplayLevel)
	{

		// create and configer StructureLearner with table
		StructureLearner structLearner = new StructureLearner();
		structLearner.setProgressInfoDisplayLevel(
								progressInfoDisplayLevel );

		CardinalityLearner cardLearner = structLearner.getCardLearner();
		cardLearner.setProgressInfoDisplayLevel( 0 );

		ParameterLearner paraLearner = cardLearner.getParameterLearner();
		paraLearner.setNumOfStartPoints( 32 );
		paraLearner.setStopThreshold( 0.01 );
		paraLearner.setProgressInfoDisplayLevel( 0 );


		// specify directory for itermediate output
		structLearner.setOutputDirectory( outputDirectory );
		cardLearner.setOutputDirectory( outputDirectory );
		paraLearner.setOutputDirectory( outputDirectory );

		structLearner.showSettings(new PrintWriter(System.out));

		// learn model
		HNB  learnedModel = structLearner.hillClimb(table);
		if (progressInfoDisplayLevel>0)
		{
		 System.out.println("Structure learning completed");
		 System.out.println("New refine parameters");
	 	}

		// refine parameters
		paraLearner.setStopThreshold( 0.0001 );
		learnedModel = paraLearner.decomposeEM( learnedModel, table);

		return learnedModel;

	}

	/**
	  * Learn NaiveBayes model from data file.
	  */
	public  static HNB learnNBModelFromFile( String dataFile )
	{
		LabelledDataTable table = (LabelledDataTable)
								LabelledDataTable.readData( dataFile );
		return learnNBModel(table );
	}
	/**
	  * Learn a naive Bayes model (a special LSB model)
	  * from the data table provided.
	  */
	public static HNB learnNBModel( LabelledDataTable table)
	{
		StructureLearner structLearner = new StructureLearner();
		return structLearner.learnNBModel( table.getHeader().copy(),table);
	}


	/**
	  * Classify record from table.  Variables in the table header
	  * must match those in this LSB model, with the
	  * class variable appear last.
	  */
	public int classify( DataTable.Record record, LabelledDataTable table)
	{

		// compute posterior of root variable. not normalized
		Node root = getRoot();
		Function1V resultFun = root.getPriorProb().copy();

		Iterator iter = root.getChildren().iterator();
		while (iter.hasNext() )
		{
			Node child = (Node) iter.next();
			Function1V tmpFun = message( child, root, record, table.getHeader());
			resultFun.multiply( tmpFun );
		}

		resultFun.normalize();

		// find the value of root variable that
		// has the maximum posterior prob
		int classLabel = -1;
		double bestProb = 0.0;

		NumberFormat formatter = NumberFormat.getNumberInstance();
		formatter.setMaximumFractionDigits(4);
		formatter.setMinimumFractionDigits(4);

		for (int i=0; i<root.getVar().getSize(); i++)
		{
		//	System.out.print( formatter.format(resultFun.getVal(i)) + " ");
			if ( resultFun.getVal(i) > bestProb )
			{
					bestProb = resultFun.getVal(i);
					classLabel = i;
			}

		}

	/*	System.out.println( "best " + formatter.format(bestProb) + " "
							+ "c=" + classLabel
							+ " c0=" + table.classLabelOfRecord(record)  );

	*/
		return classLabel;
	}
	/**
	  *  Read model from data. The returned object is
	  *  actually an HNB object.
	  */
	public static HLCM readModel( String fileName )
	{
		HLCM hlcm = HLCM.readModel(fileName);

		HNB model = new HNB( hlcm.getRoot() );

		return model;
	}
  }
