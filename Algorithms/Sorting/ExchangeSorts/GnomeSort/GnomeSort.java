package Sort.ExchangeSorts.GnomeSort;

public class GnomeSort {
	
	public static void gnomeSort_SmallToLarge(int[] arr) {
		int position = 0;
		while(position < arr.length) {
			if(position == 0 || arr[position] > arr[position-1]) {
				position++;
			}
			else {
				int temp = arr[position];
				arr[position] = arr[position-1];
				arr[position-1] = temp;
				position--;
			}
		}
	}
	
	public static void gnomeSort_LargeToSmall(int[] arr) {
		int position = 0;
		while(position < arr.length) {
			if(position == 0 || arr[position] < arr[position-1]) {
				position++;
			}
			else {
				int temp = arr[position];
				arr[position] = arr[position-1];
				arr[position-1] = temp;
				position--;
			}
		}
	}

}
