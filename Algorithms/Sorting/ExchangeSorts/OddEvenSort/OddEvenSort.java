package Sort.ExchangeSorts.OddEvenSort;

public class OddEvenSort {
	
	/*
	public static void oddEvenSort_SmallToLarge(int[] arr) {
		boolean sorted = false;
		while(!sorted)
		{
			for(boolean odd_even = false;;odd_even = !odd_even) {// false==even true==odd
				
			}
		}
	}
	*/
	
	public static void oddEvenSort_SmallToLarge(int[] arr) {
		boolean sorted = false;
		while(!sorted)
		{
			sorted = true;
			for(int odd_even = 0; odd_even < 2; odd_even++) {
				for(int i = odd_even; i < arr.length - 1; i += 2) {
					if(arr[i] > arr[i+1]) {
						int temp = arr[i];
						arr[i] = arr[i+1];
						arr[i+1] = temp;
						sorted = false;
					}
				}
			}
		}
	}
	
	public static void oddEvenSort_LargeToSmall(int[] arr) {
		boolean sorted = false;
		while(!sorted)
		{
			sorted = true;
			for(int odd_even = 0; odd_even < 2; odd_even++) {
				for(int i = odd_even; i < arr.length - 1; i += 2) {
					if(arr[i] < arr[i+1]) {
						int temp = arr[i];
						arr[i] = arr[i+1];
						arr[i+1] = temp;
						sorted = false;
					}
				}
			}
		}
	}

}
