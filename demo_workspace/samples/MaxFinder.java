import java.util.Arrays;

public class MaxFinder {
    public static int findMax(int[] numbers) {
        if (numbers == null || numbers.length == 0) {
            throw new IllegalArgumentException("Array must not be null or empty");
        }
        int max = numbers[0]; // Initialize max to the first element of the array
        for (int i = 1; i < numbers.length; i++) {
            if (numbers[i] > max) {
                max = numbers[i];
            }
        }
        return max;
    }

    public static void main(String[] args) {
        int[] numbers = {3, 5, 1, 8, 2};
        int max = findMax(numbers);
        System.out.println("The maximum number is: " + max);
    }
}