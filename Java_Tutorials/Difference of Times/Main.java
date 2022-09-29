import java.util.Scanner;

class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int secConv = 3600;
        int seco = 60, res;
        // First token starts
        System.out.println("Enter a integer value for first moment, Hour: ");
        int firstHour = scanner.nextInt();
        System.out.println("Enter a integer value for first moment, Minute: ");
        int firstMinute = scanner.nextInt();
        System.out.println("Enter a integer value for first moment, Seconds: ");
        int firstSecond = scanner.nextInt();
            //Second Token starts
            System.out.println("Enter a integer value greater than first moment; for second moment, Hour: ");
            int secondHour = scanner.nextInt();
            System.out.println("Enter a integer value for second moment, Minute: ");
            int secondMinute = scanner.nextInt();
            System.out.println("Enter a integer value for second moment, Seconds: ");
            int secondSecond = scanner.nextInt();
                //
                int _firstHourToseconds = firstHour * secConv; 
                //  _firstHourToseconds which consists -
                //values of "seconds" by using the formula of : seconds = hour * 3600.
                int _firstMinToseconds = firstMinute * seco;
                //  _firstMinToseconds which stores -
                //values of "seconds" by using the formula of : seconds = minute * 60.
                int _secondHourToseconds = secondHour * secConv;
                //  _secondHourToseconds which stores -
                //values of "seconds" by using the formula of : seconds = hour * 3600.
                int _secondMinToseconds = secondMinute * seco;
                //  _secondMinToseconds which stores -
                //values of "seconds" by using the formula of : seconds = minute * 60.
                //System.out.print(_secondMinToseconds);
                int res1 =_firstHourToseconds + _firstMinToseconds + firstSecond;      
                int res2 =_secondHourToseconds + _secondMinToseconds + secondSecond;
                res = res2 - res1 ;
                System.out.println("Seconds between these two moments of time is: ");
                System.out.println(res);
        
        
    }
}
