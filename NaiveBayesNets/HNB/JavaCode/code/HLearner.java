/**
 * @version 1.00  January 2002
 * @author Nevin L. Zhang
 */


import java.util.*;
import java.io.*;
import hlcm.*;
import hnb.*;

/**
  *  Test HeuristicLearner
  */
public class HLearner
{
	public static void main( String args[] )
	{
		if ( args.length <1 )
		{
			System.out.println("Usage: java HeuristicLearner dataFile");
			System.exit(1);
		}


		HNB model = HeuristicLearner.learnHNBFromFile( args[0] );

		model.printToFile( "HNBHeuristic.txt");
	}


}