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
public class SplitDataTable
{
	public static void main( String args[] )
	{
		if ( args.length <1 )
		{
			System.out.println("Usage: java SplitDataTable  dataFile ratio%");
			System.exit(1);
		}

		LabelledDataTable table = (LabelledDataTable)LabelledDataTable.readData( args[0]);

		int percent = Integer.parseInt( args[1] );
		DataTable[] tables = table.split( percent/100.0);

		tables[0].printToFile("tmp1.data");
		tables[1].printToFile("tmp2.data");
	}


}