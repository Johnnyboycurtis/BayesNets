/**
 * @version 1.00  January 2002
 * @author Nevin L. Zhang
 */


import java.util.*;
import java.io.*;
import hlcm.*;
import hnb.*;
import hnb.StructureLearner;

/**
  *  NaiveBayes Learner
  */
public class NBLearner
{
	public static void main( String args[] )
	{
		if ( args.length <1 )
		{
			System.out.println("Usage: java HeuristicLearner dataFile");
			System.exit(1);
		}

		HNB model =HNB.learnNBModelFromFile( args[0] );

		model.printToFile( "NaiveBayes.txt");
	}


}