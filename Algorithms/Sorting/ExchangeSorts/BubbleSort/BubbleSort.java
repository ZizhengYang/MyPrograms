package Sort.ExchangeSorts.BubbleSort;

public class BubbleSort {
	
	public static void bubbleSort_SmallToLarge(int[] arr)
	{
		int temp = 0;
		int len = arr.length;
		for(int i = 0; i < len - 1; i++) {
			for(int j = 0; j < len - 1 - i; j++) {
				if(arr[j] > arr[j+1]) {
					temp = arr[j];
					arr[j] = arr[j+1];
					arr[j+1] = temp;
				}
			}
		}
	}
	
	public static void bubbleSort_LargeToSmall(int[] arr)
	{
		int temp = 0;
		int len = arr.length;
		for(int i = 0; i < len - 1; i++) {
			for(int j = 0; j < len - 1 - i; j++) {
				if(arr[j] < arr[j+1]) {
					temp = arr[j];
					arr[j] = arr[j+1];
					arr[j+1] = temp;
				}
			}
		}
	}

}
