

/**
 * @version 1.00  Feb 2002
 * @author Nevin L. Zhang
 */

package hnb;

import hlcm.*;
import java.io.*;
import java.util.*;
import cern.jet.stat.*;

/**
  * This class implements heuristics for constructing
  * HNB models from data
  *
  */
public class HeuristicLearner
{


	/**
	  * Tester for this class
	  */
	public static void main( String args[] )
	{
		if ( args.length <1 )
		{
			System.out.println("Usage: java HeuristicLearner dataFile");
			System.exit(1);
		}

		LabelledDataTable table = (LabelledDataTable)
								LabelledDataTable.readData( args[0] );

		Variable.List varList = table.getHeader();
		int N = varList.size();

		for (int i=0; i<N-1; i++)
		{
			Variable var1 = varList.getVar(i);
			for (int j=i+1; j<N-1; j++)
			{
				Variable var2 = varList.getVar(j);

				if ( !HeuristicLearner.isIndependent( table, var1, var2 ) )
					System.out.print( "[" + var1.getName() + ", "
										 + var2.getName()+  "] " );
			}
		}

		System.out.println();
	}

	/**
	  * Learn HNB model from data file using the heuristics described
	  * in the uai02 paper.
	  */
	public static HNB learnHNBFromFile( String dataFile )
	{
		LabelledDataTable table = (LabelledDataTable)
								LabelledDataTable.readData( dataFile );
		return learnHNB( table );
	}

	/**
	  * Learn HNB model using the heuristics described
	  * in the uai02 paper.
	  */
	public static HNB learnHNB( LabelledDataTable table )
	{
		// ------- construct model structure
		// root
		Variable rootVar = table.getLabelFieldVar().copy();
		Node root = new Node( rootVar);
		HNB modelStructure = new HNB( root );

		// partition feature variables using conditional
		// independence test
		ArrayList partition = partitionFeatures( table );

		// consider the partitions one by one
		Iterator iter = partition.iterator();
		while ( iter.hasNext() )
		{
			Variable.List varList = (Variable.List) iter.next();

			if (varList.size() ==1)
			{
				// no need to introduce a latent variable
				Node node = new Node( varList.getVar(0).copy() );
				node.makeParent( root );
			}
			else
			{
				// need to create a latent variable
				Node latentNode = new Node( new Variable( 2 ) );
				latentNode.makeParent( root);

				for (int i=0; i<varList.size(); i++)
				{
					Node node = new Node( varList.getVar(i).copy() );
					node.makeParent(latentNode);
				}
			}

		}


		// learn cardinality for latent nodes and parameters
		CardinalityLearner cardLearner = new CardinalityLearner();
		HNB model = cardLearner.decomposeHillClimb(modelStructure, table);

		ParameterLearner paraLearner = new ParameterLearner();
		paraLearner.setStopThreshold( 0.0001 );
		model = paraLearner.decomposeEM( model, table);

		return model;
	}


	/**
	  * Partition feature variables in table so that
	  * variables in different partition are conditionally
	  * independent given the class variable.
	  */
	 static ArrayList partitionFeatures(LabelledDataTable table )
	{

		Variable.List varList = table.getHeader();
		int N = varList.size()-1;  // number of features

		// build the matrix representation of the graph where
		// two nodes are connected iff there are not conditional
		// independent given the class varibale.
		boolean[][] connected = new boolean[N][N];
		for (int i=0; i<N; i++)
		{
			Variable var1 = varList.getVar(i);
			for (int j=i+1; j<N; j++)
			{
				Variable var2 = varList.getVar(j);

				if ( HeuristicLearner.isIndependent( table, var1, var2 ) )
					connected[i][j]= connected[j][i] = false;
				else
					connected[i][j]= connected[j][i] = true;
			}
		}


		// transitive closure of the matrix
		boolean done = false;
		boolean[][] A = new boolean[N][N]; // work space
		do {
			done = true;
			for (int i=0; i<N; i++)
			{	for (int j=i+1; j<N; j++)
				{
					A[i][j] = A[j][i] = connected[i][j];

					if (A[i][j] == false )
					{
						// see if we can find a bridge between i and j
						for (int k=0; k<N; k++)
						{
							// here we need symmetry
							if (  connected[i][k]  && connected[j][k]  )
								A[i][j] = A[j][i]= true;
						}
					}
				}
			}

			// copy information from A to connected
			for (int i=0; i<N; i++)
			{	for (int j=i+1; j<N; j++)
				{
					// process will continue if
					// A differs from connected
					if ( connected[i][j] != A[i][j] )
						done = false;

					connected[i][j]  = connected[j][i] = A[i][j];
				}
			}

		} while ( !done );

		// ***  build partion
		ArrayList partition = new ArrayList();

		// use an array to keep track the nodes yet to be added
		// clusters
		boolean[] clustered = new boolean[N];
		for (int i=0; i<N; i++)
			clustered[i] = false;


		for (int i=0; i<N; i++)
		{
			// if node i has not been added to any cluster
			// creat a new cluster
			if ( clustered[i] == false )
			{
				Variable.List cluster = new Variable.List();
				partition.add( cluster );

				Variable var1 = varList.getVar(i);
				cluster.add( var1.copy() );

				for (int j=i+1; j<N; j++)
				{
					if ( connected[i][j] )
					{
						Variable var2 = varList.getVar(j);
						cluster.add( var2.copy() );

						clustered[j] = true; // important
					}
				}
			}
		}

		return partition;
	}


	/**
	  * Test where two varibles var1 and var2 in a LabelledDataTable are
	  * conditionally  independent given the class variable.
	  * @param table: An labelledDataTable.
	  * @param var1: A feature variable.
	  * @param var2: Another feature variable.
	  * @return true iff the two variables ARE conditionally independent
	  *         Given the class variables.
	  *         This is determined by comparing the empirical three-way
	  *		    frequencies with the theoretic frequencies derived
	  *		    assuming independence.
	  *			using GSquared test.  The variables are  conditional independent
	  *			iff the p-value is greater than 0.05 (in the sense that
	  * 		there isn't sufficient evidence to suggest otherwise.)
	  *
	  */
	public static boolean isIndependent(LabelledDataTable table , Variable var1, Variable var2 )
	{
		// project the table onto var1, var2, and classVar
		Variable.List varList = new Variable.List();
		varList.add( var1.copy() );
		varList.add( var2.copy() );

		Variable classVar = table.getLabelFieldVar();
		varList.add( classVar.copy() );

		LabelledDataTable empiricalTab = (LabelledDataTable) table.project( varList );
		double W = empiricalTab.getTotalWeight();

		Function empiricalFun = new Function( empiricalTab);

		// projection onto classVar and var1
		varList = new Variable.List();
		varList.add( classVar.copy() );
		varList.add( var1.copy() );
		Function fun1 =  empiricalFun.project( varList );


		// projection onto classVar and var2
		varList = new Variable.List();
		varList.add( classVar.copy() );
		varList.add( var2.copy() );
		Function fun2 =  empiricalFun.project( varList ) ;

		// multiplication of fun1 and fun2
		Function fun = fun1.multiply( fun2 );


		// marginal on class variable
		varList = new Variable.List();
		varList.add( classVar.copy() );
		Function fun3 = empiricalFun.project(varList );


		// theoretical frequencies
		fun = fun.divide( fun3 );
		DataTable theoreticalTab = fun.toDataTable();

		// synchronize variable order !!
		empiricalTab.synchronizeVarOrder( theoreticalTab);

		//empiricalTab.printToFile("emp.txt");
		//theoreticalTab.printToFile("theo.txt");

		// compute GSquared:
		//      S
		//     SUM 2 f(s) ln [f(s)/e(s)]
		//     s=1
		//
		//Where:
		//
		//     s      indexes response patterns
		//     S    = total number of different observed response patterns
		//     f(s) = the observed frequency of response pattern s
		//     e(s) = the expected frequency of response pattern s

		double N = theoreticalTab.getSize();
		double gSquared =0.0;
		int m=0;		// record index for empirical distribution
					    // this theoretical distribution
					    // should no fewer records than the empirical
		for (int n=0; n<N; n++)
		{
			DataTable.Record recordE = theoreticalTab.getRecord( n );
			DataTable.Record recordS = empiricalTab.getRecord( m );

			// if this condition is not true, then we don't have
			// an entry in the empirical table that corresponds to recordE
			if ( recordE.equals( recordS) )
			{
				m++;


				double es = recordE.getWeight();
				double fs = recordS.getWeight();

				if ( es >= HLCM.ZERO )
					gSquared += 2.0 * fs * Math.log( fs / es );

			}
		}

		// compute p-value:  Is the the correct degree of freedom.
		int df = var1.getSize() * var2.getSize()*classVar.getSize() - 1;

		double pval = Probability.chiSquareComplemented(df, gSquared );

		//--System.out.print(" [df=" + df + " gS=" + gSquared
		//--						+ " pv=" + pval+"]");

		// If the two variables are conditionally independent, the
		// gSqaured shoul be small, or equivalently if the pvalue should be large.
		if (pval >= 0.05)
			return true;
		else
			return false;
	}
}
