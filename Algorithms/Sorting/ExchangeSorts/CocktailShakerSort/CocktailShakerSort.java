package Sort.ExchangeSorts.CocktailShakerSort;

public class CocktailShakerSort {
	
	public static void CocktailShakerSort_SmallToLarge(int[] arr){
		
		int temp = 0;
		int left = 0, right = arr.length;
		while(left < right)
		{
			for(int i = left; i < right - 1; i++) {
				if(arr[i] > arr[i+1]) {
					temp = arr[i];
					arr[i] = arr[i+1];
					arr[i+1] = temp;
				}
			}
			right--;
			for(int i = right - 2; i >= left; i--) {
				if(arr[i] > arr[i+1]) {
					temp = arr[i];
					arr[i] = arr[i+1];
					arr[i+1] = temp;
				}
			}
			left++;
		}
		
	}
	
	public static void CocktailShakerSort_LargeToSmall(int[] arr){

		int temp = 0;
		int left = 0, right = arr.length;
		while(left < right)
		{
			for(int i = left; i < right - 1; i++) {
				if(arr[i] < arr[i+1]) {
					temp = arr[i];
					arr[i] = arr[i+1];
					arr[i+1] = temp;
				}
			}
			right--;
			for(int i = right - 2; i >= left; i--) {
				if(arr[i] < arr[i+1]) {
					temp = arr[i];
					arr[i] = arr[i+1];
					arr[i+1] = temp;
				}
			}
			left++;
		}

	}

}
