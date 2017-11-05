
/**
 * @version 1.00  January 2002
 * @author Nevin L. Zhang
 */

import hlcm.*;
import hnb.*;
import hnb.ScoreCalculator;
import hnb.ParameterLearner;
import hnb.CardinalityLearner;
import hnb.StructureLearner;
import hnb.Evaluator;
import hnb.Utility;
import exp.*;
import java.util.*;
import java.io.*;
import cern.jet.stat.*;


public class Main
{
	public static void main( String[] args )
	{


		if ( args.length  < 1)
		{
			System.out.println("Usage: java Main action parameters ");
			System.exit(1);
		}


		String todo = args[0];

		if (todo.equals( "learn" ) )
		{
			if ( args.length  < 3 )
			{
				System.out.println("Usage: java Main learn " +
									"dataFileName  ouputDirectory");
				System.exit(1);
			}
			String fileName= args[1];
			String directory= args[2];

			// read data
			LabelledDataTable table = (LabelledDataTable)LabelledDataTable.readData( fileName );


			// This is to avoid variable name conflict
			// This is only necessary if the data is synthetic.
			// In that case, there could be name conflit
			for (int i=0; i<200; i++)
				new Variable( 2 );

			HNB learnedModel =
				HNB.learnHNB( table, directory, 3);

			learnedModel.printToFile(directory+"/ModelFinal.txt");


		}
		else if (todo.equals( "evaluateCV" ) )
		{
			if ( args.length  < 3 )
			{
				System.out.println("Usage: java Main evaluateCV " +
									"learningMethod dataFileName outputDirctory");
				System.exit(1);
			}
			String learningMethod = args[1];
			String dataFileName= args[2];
			String outputDirectory = args[3];

			Evaluator eval = new Evaluator();

			double accuracy = eval.crossValidation( learningMethod, dataFileName,
								10, outputDirectory);

			// read data
			LabelledDataTable table = (LabelledDataTable)LabelledDataTable.readData( dataFileName );

			double confidence = eval.confidenceInterval( accuracy, (int) table.getTotalWeight() );

			System.out.println("\n Performace:" + accuracy + "=-" + confidence );
		}
		else if (todo.equals( "evaluate" ) )
		{
			if ( args.length  < 3 )
			{
				System.out.println("Usage: java Main evaluate " +
									"learningMethod  trainData testData outputDirctory");
				System.exit(1);
			}
			String learningMethod = args[1];
			String trainData= args[2];
			String testData= args[3];
			String outputDirectory = args[4];


			// read data
			LabelledDataTable trainTable =
						(LabelledDataTable)LabelledDataTable.readData( trainData );

			HNB learnedModel = null;
			if ( learningMethod.equals("HNB") )
			   learnedModel =	HNB.learnHNB( trainTable,
									outputDirectory +"/" + learningMethod, 2);
			else
			 learnedModel =	HNB.learnNBModel( trainTable);

			learnedModel.printToFile( outputDirectory + "/" + learningMethod
										+ "/FinalModel.txt");

			LabelledDataTable testTable =
						(LabelledDataTable)LabelledDataTable.readData( testData );
			learnedModel.synchronizeVarIDs( testTable.getHeader() );

			Evaluator eval = new Evaluator();


			double accuracy = eval.classificationAccuracy(learnedModel, testTable);


			double confidence = eval.confidenceInterval( accuracy, (int)
						testTable.getTotalWeight() );

			System.out.println("\n Performace:" + accuracy + "=-" + confidence );
		}
		else if (todo.equals( "evaluateModel" ) )
		{
			if ( args.length  < 3 )
			{
				System.out.println("Usage: java Main evaluateModel " +
									"modelFileName  testData");
				System.exit(1);
			}
			String modelFile = args[1];
			String testData= args[2];


			HNB model = (HNB) HNB.readModel( modelFile );




			LabelledDataTable testTable =
						(LabelledDataTable)LabelledDataTable.readData( testData );
			model.synchronizeVarIDs( testTable.getHeader() );

			Evaluator eval = new Evaluator();


			double accuracy = eval.classificationAccuracy(model, testTable);


			double confidence = eval.confidenceInterval( accuracy, (int)
						testTable.getTotalWeight() );

			System.out.println("\n Performace:" + accuracy + "=-" + confidence );
		}
		else
		{
			System.out.println(" action option not available");
		}


	}

}



/*



		--- test for labelled data table
		LabelledDataTable table = (LabelledDataTable)
				LabelledDataTable.readData( "data/breastCancer.data" );


		DataTable[] partition = table.split( 0.2);


		partition[0].printToFile( "data/tmp0.data");
		partition[1].printToFile( "data/tmp1.data");

		LabelledDataTable t = (LabelledDataTable) partition[0];
		LabelledDataTable table2 = t.merge((LabelledDataTable) partition[1] );
		table2.printToFile( "data/tmp3.data");

		--- test method of HNB

		Variable.List varList = model.getObservedVariables();
		varList.printToFile("tmpObs.txt");

		Node.List nodeList = model.hiddenNodes();
		nodeList.printToFile("tmpHid.txt");

		HNB[] modelList = model.decompose();

		for (int i=0; i<modelList.length; i++)
		{
			modelList[i].printToFile("tmpModel."+i+".txt");
		}

		model = HNB.merge(modelList);

		model.printToFile("tmpModel.merged.txt");

		DataTable.Record record = table.getRecord(0);

		record.setCell(7, 0);
		double a= model.probOfDatum( record, table ) ;
		System.out.println( a);

		record.setCell(7, 1);
		double b= model.probOfDatum( record, table ) ;
		System.out.println( b);


		record.setCell(7, 2);
		double c= model.probOfDatum( record, table ) ;
		System.out.println( c);



		System.out.println( a + b + c);

		HLCM hModel = (HLCM) model;

		System.out.println( hModel.probOfDatum( record, table.getHeader() ) );



		ParameterLearner pLearner = new ParameterLearner();
		pLearner.setProgressInfoDisplayLevel(1);
		pLearner.setStopThreshold(0.01);
		pLearner.showSettings( new PrintWriter (System.out));

		HNB newModel = pLearner.decomposeEM(model, table);

	StructureLearner structLearner = new StructureLearner();
		ArrayList models = structLearner.neighborModels( model );

		int i=0;
		Iterator iter = models.iterator();
		while ( iter.hasNext() )
		{
			HNB m = (HNB) iter.next();
			m.printToFile("nextModel." + i + ".txt");
			i++;
		}


String directory = "models";

		HNB model = Generator.createHNB();
		model.printToFile( "models/MOrg.txt");


		Generator gen = new Generator();

		LabelledDataTable table = gen.createTableFromHNB(model, 10000);
		table.printToFile("models/Table.txt");


		// preprocess uci data

		LabelledDataTable t = (LabelledDataTable)
									LabelledDataTable.readData( args[0], "class");
		t.printToFile("data/tmp.txt");

		System.exit(1);
		if ( args.length  < 1)
		{
			System.out.println("Usage: java Main action parameters ");
			System.out.println("\"action\" might be: \"learn\", \"evaluate\" ");
			System.exit(1);
		}


		// manipulate data

		String file = args[0];
		LabelledDataTable tmpT = (LabelledDataTable)LabelledDataTable.readData( file);
		for (int i =0; i<tmpT.getSize(); i++)
		{
			DataTable.Record r = tmpT.getRecord(i);

			r.setWeight( 20.0);
		}
		tmpT.printToFile( file );

		System.exit(1);


if (1==1)
	  {
		// testing handling of missing data
		HNB m = Generator.createHNB();
		m.printToFile("tmpModel.txt");

		Variable.List varList = m.getObservedVariables();
		LabelledDataTable t = new LabelledDataTable( varList );

		int n = varList.size();

		DataTable.Record r1 = new DataTable.Record( varList.size() );
		r1.setCell(n-1, 0);
		t.addRecord( r1 );

		DataTable.Record r2 = new DataTable.Record( varList.size() );
		r2.setCell(n-1, 1);
		t.addRecord( r2 );
		DataTable.Record r3 = new DataTable.Record( varList.size() );
		r3.setCell(n-1, 2);
		t.addRecord( r3 );
	for (int i=0; i<100; i++)
	{
		r3 = new DataTable.Record( varList.size() );
		r3.setCell(n-1, 2);
		r3.setCell(0, 0);
		t.addRecord( r3 );
	}
		t.printToFile("tmpTable.txt");

		LabelledDataTable t1 = (LabelledDataTable)
				LabelledDataTable.readData("tmpTable.txt");
		t1.printToFile("tmpTable1.txt");
		m.synchronizeVarIDs( t1.getHeader() );

		System.out.println( m.probOfDatum(
					t1.getRecord(0), t1 ));

		System.out.println( m.probOfDatum(
					t1.getRecord(1), t1) );
		System.out.println( m.probOfDatum(
				t1.getRecord(2), t1 ) );

		ParameterLearner l = new ParameterLearner();
	for (int i=0; i<100; i++)
		m = l.oneStepEM(m, t1);
		m.printToFile("tmpModel1.txt");


		System.exit(1);
      }

		if (1==1)
		{
			HNB model = Generator.createHNB();
					model.printToFile( "MOrg.txt");

			StructureLearner structLearner = new StructureLearner();
			ArrayList models = structLearner.neighborModels( model );

			int i=0;
			Iterator iter = models.iterator();
			while ( iter.hasNext() )
			{
				HNB m = (HNB) iter.next();
				m.printToFile("nextModel." + i + ".txt");
				i++;
			}

			System.exit(1);
		}



		if (1==1)
				{
					// sample model

					String modelFile = args[0];
					String outputDir = args[1];

					HNB model = (HNB) HNB.readModel(modelFile);


					model.printToFile("tmpModel.txt");
					LabelledDataTable table = null;


					table = Utility.sampleHNB( model, 1000);
					table.printToFile( outputDir +"/data.1000");


					table = Utility.sampleHNB( model, 5000);
					table.printToFile( outputDir +"/data.5000");


					table = Utility.sampleHNB( model, 10000);
					table.printToFile( outputDir +"/data.10000");


					table = Utility.sampleHNB( model, 50000);
					table.printToFile( outputDir +"/data.50000");


					table = Utility.sampleHNB( model, 100000);
					table.printToFile( outputDir +"/data.100000");


					System.exit(1);
				}

*/
