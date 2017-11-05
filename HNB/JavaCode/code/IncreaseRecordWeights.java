/**
 * @version 1.00  January 2002
 * @author Nevin L. Zhang
 */

import hlcm.*;
import hnb.*;
import java.io.*;


public class IncreaseRecordWeights
{
	public static void main( String[] args )
	{
		if (args.length <3)
		{
			System.out.println("Usage: java IncreaseRecordWeights inputfile folds outputfile");

			System.exit(1);
		}

		LabelledDataTable t = (LabelledDataTable)
					LabelledDataTable.readData(args[0]);

		int n = Integer.parseInt( args[1] );

		for (int i=0; i<t.getSize(); i++)
		{
			DataTable.Record record = t.getRecord(i);

			record.setWeight( record.getWeight() * n );
		}
		t.printToFile( args[2]);


	}
}
