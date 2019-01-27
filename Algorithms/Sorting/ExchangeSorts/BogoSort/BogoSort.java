package Sort.ExchangeSorts.BogoSort;

import java.util.Random;

public class BogoSort {
	
	static Random random = new Random();
	
	public static void bogoSort(int[] n) {
	    while(!inOrder(n))shuffle(n);
	}

	public static void shuffle(int[] n) {
	    for (int i = 0; i < n.length; i++) {
	        int swapPosition = random.nextInt(i + 1);
	        int temp = n[i];
	        n[i] = n[swapPosition];
	        n[swapPosition] = temp;
	    }
	}

	public static boolean inOrder(int[] n) {
	    for (int i = 0; i < n.length-1; i++) {
	        if (n[i] > n[i+1]) return false;
	    }
	    return true;
	}

}
