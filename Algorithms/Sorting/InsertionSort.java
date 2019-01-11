package Sort.InsertionSort;

import java.util.Arrays;

public class InsertionSort {
	
	public static void insertionSortSmallToLarge(int[] arr){
		
		for(int i = 1; i < arr.length; i++){
		
			int pivot = arr[i];
			int j = i - 1;
			while(j >= 0 && arr[j] > pivot){
				arr[j+1] = arr[j];
				j--;
			}
			arr[j+1] = pivot;
			
		}
		
	}
	
	public static void insertionSortLargeToSmall(int[] arr){
		
		for(int i = 1; i < arr.length; i++){
		
			int pivot = arr[i];
			int j = i - 1;
			
			while(j >= 0 && arr[j] < pivot){
				arr[j+1] = arr[j];
				j--;
			}
			arr[j+1] = pivot;
			
		}
		
	}

}
