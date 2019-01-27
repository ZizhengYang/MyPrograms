public class CombSort {
	
	public static double declining_Rate = 1.3;
	
	public static void combSort_SmallToLarge(int[] arr) {
		boolean sorted = false;
		int gap = arr.length;
		while(gap > 1 || !sorted) {
			if(gap > 1) {
				gap = ((int)(gap / declining_Rate));
			}
			sorted = true;
			for(int i = 0; (i + gap) < arr.length; i++) {
				if(arr[i] > arr[i+gap]) {
					int temp = arr[i];
					arr[i] = arr[i+gap];
					arr[i+gap] = temp;
					sorted = false;
				}
			}
		}
	}
	
	public static void combSort_LargeToSmall(int[] arr) {
		boolean sorted = false;
		int gap = arr.length;
		while(gap > 1 || !sorted) {
			if(gap > 1) {
				gap = ((int)(gap / declining_Rate));
			}
			sorted = true;
			for(int i = 0; (i + gap) < arr.length; i++) {
				if(arr[i] < arr[i+gap]) {
					int temp = arr[i];
					arr[i] = arr[i+gap];
					arr[i+gap] = temp;
					sorted = false;
				}
			}
		}
	}

}
