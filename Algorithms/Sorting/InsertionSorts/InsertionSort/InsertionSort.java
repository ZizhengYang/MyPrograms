package Sort.InsertionSort.InsertionSort;

/**
 * 
 * <table>
 * 		<tr>
 * 			<th>Worst-case performance: &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp</th>
 * 			<th>n^2</th>
 * 		</tr>
 * 		<tr>
 * 			<th>Best-case performance: &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp</th>
 * 			<th>n</th>
 * 		</tr>
 * 		<tr>
 * 			<th>Average performance: &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp</th>
 * 			<th>n^2</th>
 * 		</tr>
 * 		<tr>
 * 			<th>Worst-case space complexity: &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp</th>
 * 			<th>n</th>
 * 		</tr>
 * </table>
 *
 */
public class InsertionSort {

	public static void insertionSort_SmallToLarge(int[] arr){

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
	
	public static void insertionSort_LargeToSmall(int[] arr){
		
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
