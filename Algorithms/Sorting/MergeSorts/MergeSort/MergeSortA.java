package Sort.MergeSort.MergeSort;

public class MergeSort {
	
	// implement the algrithm by recursion

	public static void doMergeSort_SmallToLarge(int[] arr) {

		MergeSort.Sort_SmallToLarge(arr, 0, arr.length-1);

	}

	public static void doMergeSort_LargeToSmall(int[] arr) {

		MergeSort.Sort_LargeToSmall(arr, 0, arr.length-1);

	}

	public static void Sort_SmallToLarge(int[] arr, int start, int end) {

		if(end == start) {return;}

		int sortSize = end - start;
		int mid = start + sortSize / 2;

		Sort_SmallToLarge(arr, start, mid);
		Sort_SmallToLarge(arr, mid+1, end);

		Merge_SmallToLarge(arr, start, end, mid);

	}

	public static void Sort_LargeToSmall(int[] arr, int start, int end) {

		if(end == start) {return;}

		int sortSize = end - start;
		int mid = start + sortSize / 2;

		Sort_LargeToSmall(arr, start, mid);
		Sort_LargeToSmall(arr, mid+1, end);

		Merge_LargeToSmall(arr, start, end, mid);

	}

	public static void Merge_SmallToLarge(int[] arr, int start, int end, int mid) {

		int i = start;
		int j = mid + 1;
		int count = 0;
		int[] buffer = new int[end - start + 1];

		while(i <= mid && j <= end)
		{
			if(arr[i] >= arr[j])
			{
				buffer[count] = arr[j];
				count++;
				j++;
			}
			else if(arr[j] > arr[i])
			{
				buffer[count] = arr[i];
				count++;
				i++;
			}
		}

		while(i <= mid)
		{
			buffer[count] = arr[i];
			count++;
			i++;
		}

		while(j <= end)
		{
			buffer[count] = arr[j];
			count++;
			j++;
		}

		for(int k = start, c = 0; k <= end; k++, c++)
		{
			arr[k] = buffer[c];
		}

	}

	public static void Merge_LargeToSmall(int[] arr, int start, int end, int mid) {

		int i = start;
		int j = mid + 1;
		int count = 0;
		int[] buffer = new int[end - start + 1];

		while(i <= mid && j <= end)
		{
			if(arr[i] <= arr[j])
			{
				buffer[count] = arr[j];
				count++;
				j++;
			}
			else if(arr[j] < arr[i])
			{
				buffer[count] = arr[i];
				count++;
				i++;
			}
		}

		while(i <= mid)
		{
			buffer[count] = arr[i];
			count++;
			i++;
		}

		while(j <= end)
		{
			buffer[count] = arr[j];
			count++;
			j++;
		}

		for(int k = start, c = 0; k <= end; k++, c++)
		{
			arr[k] = buffer[c];
		}

	}

}
