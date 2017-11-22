
/**
 * @version 1.00  January 2002
 * @author Nevin L. Zhang
 */

package hnb;


/**
  *  Exception class for HNB.
  */

public class HNBException extends RuntimeException
{
  /**
	* Default constructor.
	*/
  public HNBException() {}

  /**
	* Constructor specifying message.
	*/
  public HNBException(String message)
  {
	  super( "HNBException:" + message );
  }
}