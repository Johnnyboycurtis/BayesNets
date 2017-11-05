

/**
 * @version 1.00  January 2002
 * @author Nevin L. Zhang
 */

package hnb;  // latent structure Bayesian models

import java.io.*;
import java.util.*;
import cern.jet.random.*;  // this package has a better random
							// number generator
import hlcm.*;

/**
  *  LabelledDataTable is the same as Datatable except that each record
  *  is labelled.  The class variable is the last variable on the table
  *  header.  This implies that the labels are kept in the second last column.
  *  The last column contains frequency counts.
  */
public class LabelledDataTable extends DataTable
{

	public LabelledDataTable()
	{
		super();
	}

	public LabelledDataTable(Variable.List header)
	{
		super( header );

	}

	public LabelledDataTable(Variable.List header, String labelField)
	{
		super( header );
		this.labelField = labelField;
	}

	/**
	  *  Get labelField name
	  */
	public String getLabelFieldName()
	{
		return labelField;
	}

	/**
	  *  Get labelField variable
	  */
	public Variable getLabelFieldVar()
	{
		return getHeader().getVar( getHeader().size() - 1 );
	}


	/**
	  *  Set labelField
	  */
	public void setLabelFieldName(String labelField)
	{
		this.labelField = labelField;
	}

	/**
	  *  Get column index of labelField, i.e. index of the
	  *  second last column, which corresponds to the class variable
	  */
	public int getLabelFieldIndex()
	{
		return getHeader().size() - 1;
	}


	/**
	  *  Get class label of a record. For this method to
	  * work correctly, one must make sure that the class
	  * variable is the last variable on the table header.
	  */
	public int classLabelOfRecord( DataTable.Record r)
	{
		return r.getCell(getHeader().size() - 1 );
	}

	/**
	  *  Deep copy. A LabbeledDataTable object is return.
	  *  However the return type is declared to be DataTable instead of
	  *  LabelledDataTable to avoid compiling error.  Caller must
	  *  downcast the copied object to LabelledDataTable.
	  */
	public DataTable copy()
	{
		LabelledDataTable newTable =
				new LabelledDataTable( getHeader().copy(), labelField);

		for (int i=0; i<this.getSize(); i++)
		{
			newTable.addRecord( this.getRecord( i ).copy() );
		}
		return newTable;
	}



	/**
	  * Print table to a stream.
	  */
	public void show(PrintWriter out)
	{
		out.println("//Data table Produced by the HNB package");
		out.println("");
		out.println("Name: " + getName()  );
		out.println("");

		out.println("//Variables: name of variable followed by names of states");
		out.println("//           The last variable is the class variable");
		out.println("//            State names must start with English characters");


		Iterator iter = getHeader().iterator();
		while (iter.hasNext() )
		{
			Variable var = (Variable) iter.next();
			out.print(var.getName() +": " );

			for (int i=0; i< var.getSize(); i++)
			{
					out.print(var.getStateName(i) +" " );
			}
			out.println();
		}
		out.println();

		out.println("//Records: The last column contains frequencies.");
		out.println("//	        The second last column contain class labels");
		out.println("//	        0=First state, 1=second state, ...");
		out.println("//	        -1=missing value");

		for (int i=0; i< getSize(); i++)
		{
			getRecord(i).show(out);
		}

		out.println("//Total count: " + getTotalWeight() );
	}




	/**
	  *  Project table onto a subset of attributes.
	  *  The labelField must be in the subheader. Otherwise
	  *  the program terminates.
	  *  Returns a LabelledDataTable object, although the
	  *  return type has to be declared to be DataTable
	  *  to satisfy Java requirements.
	  */
	public DataTable project( Variable.List subHeader )
	{

		// First make sure that labelField is in subHeader.
		if ( subHeader.indexOf( getLabelFieldVar() ) < 0 )
		{
			throw new HNBException(
				"Error in LabelledDataTable.project: "
				+ "Class variable \""
				+ getLabelFieldVar().getName()
				+  "\" not in subHeader");

		}

		// Try to build a mapping from variables in subHeader
		// to variables in the header of this table
		int subLen = subHeader.size();
		int[] map = new int[subLen];

		for (int j=0; j<subLen; j++)
		{

			int index = getHeader().indexOf(  subHeader.getVar(j) );

			if ( index < 0 )
				throw new HNBException(
					"Error in LabelledDataTable.project: "
					+ "Project variable "
					+ subHeader.getVar(j).getID()
					+  " not fond. " + j);

			map[j] = index;
		}

		// create new Table
		LabelledDataTable newTable =
				new LabelledDataTable(subHeader, labelField);


		// set content of new table
		int N = this.getSize();  // number of records in old table

		for (int i=0; i<N; i++)
		{
			Record oldRecord = (Record) getRecord(i);

			Record newRecord = new Record( subLen );
			for (int j=0; j<subLen; j++)
			{
				newRecord.setCell(j,  oldRecord.getCell( map[j]) );
			}
			newRecord.setWeight( oldRecord.getWeight() );

			newTable.addRecord( newRecord );
		}

		return newTable;
	}


	/**
	  * Read data from file. It is assumed that the data are created by
	  * printToFile().  Among other things, this implies that the last variable
	  * on table header must be the class variable.
	  * The object returned is actually a LabelledDataTable
	  * object.
	  */
	public static DataTable readData( String fileName )
	{
		// read data and create an DataTable object
		DataTable tmpTable = DataTable.readData( fileName );

		Variable.List header = tmpTable.getHeader().copy();
		String labelField = header.getVar( header.size() - 1).getName();

		LabelledDataTable newTable = new LabelledDataTable( header, labelField);


		for (int i=0; i<tmpTable.getSize(); i++)
		{
			newTable.addRecord( tmpTable.getRecord( i ).copy() );
		}
		return newTable;
	}


	/**
	  * Read data from file.
	  * This method will shit the class variable to the last location
	  * if it is not there already.
	  * The object returned is actually a LabelledDataTable
	  * object.
	  */
	public static DataTable readData( String fileName, String labelField )
	{
		// read data and create an DataTable object
		DataTable tmpTable = DataTable.readData( fileName );

		Variable.List header = tmpTable.getHeader().copy();
		LabelledDataTable newTable = new LabelledDataTable( header, labelField);

		for (int i=0; i<tmpTable.getSize(); i++)
		{
			newTable.addRecord( tmpTable.getRecord( i ).copy() );
		}


		// now make labelField the last variable on table header
		Variable.List header1 = header.copy();
		Variable classVar = header1.findVar4Name( labelField );
		if (classVar == null )
		{
			throw new HNBException(
				"Error in LabelledDataTable.readData: "
				+ "labelField \"" + labelField +"\"not found");

		}
		header1.remove( header1.indexOf( classVar ));  //remove classVar
		header1.add( classVar ); 	// add it to the last position.

		return newTable.project( header1 );
	}


	/**
	  * Split this data table randomly into two data subtables
	  * such that the first subtable contains roughly "precentage"
	  * of the total records.
	  * This table is not altered. Weights on records
	  * must be of integer values.
	  * @para percentage: percentage of records in the
	  * first subtable.
	  */
	public DataTable[] split(double percentage)
	{
		DataTable partition[] = new DataTable[2];
		partition[0] = new LabelledDataTable( getHeader(), labelField );
		partition[1] = new LabelledDataTable( getHeader(), labelField );


		int N = getSize();
		for (int i=0; i<N; i++)
		{
			Record rec = getRecord( i );
			Record rec1 = rec.copy();
			rec1.setWeight(1.0);

			int w = (int) rec.getWeight();
			for (int j=0; j<w; j++)
			{
				double  p = Uniform.staticNextDouble();

				if ( p<=percentage)
				{
					partition[0].addRecord( rec1.copy() );
				}
				else
				{
				    partition[1].addRecord( rec1.copy() );
				}
			}
		}
		return partition;
	}


	/**
	  * Merge this table with another. It is assumed that the two tables
	  * have identical table headers.  The original tables
	  * are not altered. A new table is created for the
	  * result.
	  */
	public  LabelledDataTable merge( LabelledDataTable another)
	{
		LabelledDataTable newTable = (LabelledDataTable) copy();

		int N = another.getSize();
		for (int i=0; i<N; i++)
		{
			DataTable.Record r = another.getRecord(i);
			newTable.addRecord( r.copy() );
		}

		return newTable;
	}

	private String labelField = "";
}



