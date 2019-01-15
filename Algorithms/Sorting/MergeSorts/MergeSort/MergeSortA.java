package Sort.MergeSort;

public class MergeSort {

	public static void doMergeSortSmallToLarge(int[] arr) {

		MergeSort.SortSmallToLarge(arr, 0, arr.length-1);

	}

	public static void doMergeSortLargeToSmall(int[] arr) {

		MergeSort.SortLargeToSmall(arr, 0, arr.length-1);

	}

	public static void SortSmallToLarge(int[] arr, int start, int end) {

		if(end == start) {return;}

		int sortSize = end - start;
		int mid = start + sortSize / 2;

		//System.out.println(start+"--"+(mid)+"<==>"+(mid+1)+"--"+(end));
		SortSmallToLarge(arr, start, mid);
		SortSmallToLarge(arr, mid+1, end);

		MergeSmallToLarge(arr, start, end, mid);

	}

	public static void SortLargeToSmall(int[] arr, int start, int end) {

		if(end == start) {return;}

		int sortSize = end - start;
		int mid = start + sortSize / 2;

		//System.out.println(start+"--"+(mid)+"<==>"+(mid+1)+"--"+(end));
		SortLargeToSmall(arr, start, mid);
		SortLargeToSmall(arr, mid+1, end);

		MergeLargeToSmall(arr, start, end, mid);

	}

	public static void MergeSmallToLarge(int[] arr, int start, int end, int mid) {

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

	public static void MergeLargeToSmall(int[] arr, int start, int end, int mid) {

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
