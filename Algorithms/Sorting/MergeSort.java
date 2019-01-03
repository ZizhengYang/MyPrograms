package Sort.MergeSort;

public class MergeSort {
	
	public static void doMergeSort(int[] arr) {
		
		MergeSort.Sort(arr, 0, arr.length);
		
	}
	
	public static void Sort(int[] arr, int start, int end) {
		
		if(start == end) {return;}
		
		int sortSize = end - start + 1;
		int mid;
		//if (sortSize % 2 == 0) {
		//	mid = start + sortSize / 2 - 1;
		//} else {
			mid = start + sortSize / 2;
		//}
		
		Sort(arr, start, mid);
		Sort(arr, mid + 1, end);
		
		Merge(arr, start, end, mid);
		
	}
	
	public static void Merge(int[] arr, int start, int end, int mid) {
		
		int i = start;
		int j = mid + 1;
		int count = 0;
		int[] buffer = new int[arr.length];
		
		while(i < mid + 1 && j < end) {
			
			if(arr[i] >= arr[j]) {
				
				buffer[count] = arr[j];
				i++;
				count++;
				
			}
			
			if(arr[i] < arr[j]) {
				
				buffer[count] = arr[i];
				j++;
				count++;
				
			}
			
		}
		
		while(i < mid + 1) {
			
			buffer[count] = arr[i];
			i++;
			count++;
			
		}
		
		while(j < end) {
			
			buffer[count] = arr[j];
			j++;
			count++;
			
		}
		
		arr = buffer;
		
	}

}
