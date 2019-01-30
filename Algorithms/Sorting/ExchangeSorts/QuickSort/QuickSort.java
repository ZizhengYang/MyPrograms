public class QuickSort {
	
	public static void quickSort_SmallToLarge(int[] arr) {
		QuickSort.Sort_SmallToLarge(arr, 0, arr.length);
	}
	
	public static void Sort_SmallToLarge(int[] arr, int start, int end) {
		if(start < end) {
			int pivot = partition_SmallToLarge(arr, start, end-1);
			Sort_SmallToLarge(arr, start, pivot);
			Sort_SmallToLarge(arr, pivot+1, end);
		}
	}
	
	public static int partition_SmallToLarge(int[] arr, int start, int end) {
		int pivot = arr[start];
		while(start < end) {
			while(start < end && arr[end] >= pivot) { end--; }
			arr[start] = arr[end];
			while(start < end && arr[start] <= pivot) { start++; }
			arr[end] = arr[start];
		}
		arr[start] = pivot;
		return start;
	}
	
	//////////////////////////////////////////////////
	
	public static void quickSort_LargeToSmall(int[] arr) {
		QuickSort.Sort_LargeToSmall(arr, 0, arr.length);
	}
	
	public static void Sort_LargeToSmall(int[] arr, int start, int end) {
		if(start < end) {
			int pivot = partition_LargeToSmall(arr, start, end-1);
			Sort_LargeToSmall(arr, start, pivot);
			Sort_LargeToSmall(arr, pivot+1, end);
		}
	}
	
	public static int partition_LargeToSmall(int[] arr, int start, int end) {
		int pivot = arr[start];
		while(start < end) {
			while(start < end && arr[end] <= pivot) { end--; }
			arr[start] = arr[end];
			while(start < end && arr[start] >= pivot) { start++; }
			arr[end] = arr[start];
		}
		arr[start] = pivot;
		return start;
	}

}
