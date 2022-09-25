import java.util.Scanner;

class Main {
  public static void main(String[] args) {
    Scanner userInput = new Scanner(System.in);
    int sum;
    float perc;
    System.out.println("This is the program to calculate the Percentage of scored marks of student: ");

    System.out.println("Enter marks of subject1: ");
    int a = userInput.nextInt();
    System.out.println("Enter marks of subject2: ");
    int b = userInput.nextInt();
    System.out.println("Enter marks of subject3: ");
    int c = userInput.nextInt();
    System.out.println("Enter marks of subject4: ");
    int d = userInput.nextInt();
    System.out.println("Enter marks of subject5: ");
    int e = userInput.nextInt();
    sum = a + b + c + d + e;
    perc = (sum / 100) * 100;

    System.out.println(perc + "%");
    userInput.close();
  }
}
