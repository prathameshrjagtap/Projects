import java.util.Scanner;

class Main {
  public static void main(String[] args) {
    System.out.print("Please put the km value to convert into miles: ");
    Scanner sc   = new Scanner(System.in);
    double a = sc.nextDouble();
    double b = 0.62137;
    double mile = a * b;
    System.out.println("Coversion of "+ a + "km, is "+ mile+" miles.");
  }
}
