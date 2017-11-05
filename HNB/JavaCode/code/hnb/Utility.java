

/**
 * @version 1.00  February 2002
 * @author Nevin L. Zhang
 */

package hnb;

import java.io.*;
import java.util.*;
import hlcm.*;
import cern.jet.random.*;  // this package has a better random
							// number generator

/**
  * A utility class.
  */
public class Utility
{

	/**
	  * Create a labelled table of data of observed variable from an HNB by sampling
	  */
	public  static LabelledDataTable sampleHNB( HNB model, int N )
	{
		LabelledDataTable table = new LabelledDataTable();

		/* get list of observed variables from model */
		Variable.List header = model.getObservedVariables();
		table.setHeader( header );

		/* generates N records */
		for (int i=0; i<N; i++)
		{
			DataTable.Record row = generateRow(model, header);
			table.addRecord(row);
		}

		return table;
	}


	/* ----- Begin: Private helper methods of createTableFromHLCM -----*/

	// generate one record by sampling: values of observed variables
    private static DataTable.Record generateRow(HLCM model, Variable.List header)
    {
		int len = header.size();
		DataTable.Record row = new DataTable.Record( len );

		Node node = model.getRoot();
		Function1V prob = node.getPriorProb();

		generateRowRecursion(row, node, prob, header );

		return row;
	}


    // sample all node and its descents
	private static void generateRowRecursion(
				DataTable.Record row,
				Node node,
				Function1V prob,
				Variable.List header)
	{
		Variable var = node.getVar();
		int randomVal = sample(prob);  // sample this node

		Node.List children = node.getChildren();
		int size = children.size();


		// if node has no parent, var is an observed variable
		if ( !node.hasParent() )
		{		// find position of var in header, the list of
				// of observed variables
				int index = header.indexOf( var );
				row.setCell( index, randomVal);
		}

	    // if node has not children, var is an observed variable
		if ( !node.hasChildren() )
		{		// find position of var in header, the list of
		 		// of observed variables
				int index = header.indexOf( var );
				row.setCell( index, randomVal);
		}
		else
		{
			//  recursion
			for (int i=0; i<size; i++)
			{
				Node childNode = children.getNode(i);
				Function1V probChildNode =
						childNode.getCondProb().cut( var, randomVal);

			    generateRowRecursion(row, childNode, probChildNode, header);
			}
		}
	}

	private static int sample( Function1V prob )
	{
		//double p = Math.random();
		//double p = randomNumGen.nextDouble();

		// Using cern.jet.random.Uniform
		double  p = Uniform.staticNextDouble();

		int size = prob.getSize();

		int i=0;

		double cumulation = 0.0;

		for (i=0; i<size; i++)
		{
			 cumulation += prob.getVal(i);
			 if (cumulation >= p ) return i;
		}

		return size -1;
	}

	private Random randomNumGen = new Random();

}
