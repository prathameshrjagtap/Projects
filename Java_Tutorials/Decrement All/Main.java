import java.util.Scanner;
class Main {
    public static void main(String[] args) {
        Scanner scanner= new Scanner (System.in);
        System.out.print("Enter values which you want to decrement: ");
        int numOne = scanner.nextInt();
        int numTwo = scanner.nextInt();
        int numThree = scanner.nextInt();
        int numFour = scanner.nextInt();
        numOne -= 1;
        numTwo -= 1;
        numThree -= 1;
        numFour -= 1;

        System.out.print(numOne + " ");
        System.out.print(numTwo + " ");
        System.out.print(numThree + " ");
        System.out.print(numFour + " ");
    }
}
