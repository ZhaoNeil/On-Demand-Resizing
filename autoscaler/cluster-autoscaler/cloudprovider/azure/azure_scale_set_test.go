/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package azure

import (
	"fmt"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"net/http"
	"testing"
	"time"

	apiv1 "k8s.io/api/core/v1"
	"k8s.io/autoscaler/cluster-autoscaler/cloudprovider"
	"k8s.io/legacy-cloud-providers/azure/clients/vmssclient/mockvmssclient"
	"k8s.io/legacy-cloud-providers/azure/clients/vmssvmclient/mockvmssvmclient"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-12-01/compute"
	"github.com/Azure/go-autorest/autorest/to"
	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"
)

func newTestScaleSet(manager *AzureManager, name string) *ScaleSet {
	return &ScaleSet{
		azureRef: azureRef{
			Name: name,
		},
		manager: manager,
		minSize: 1,
		maxSize: 5,
	}
}

func newTestVMSSList(cap int64, name, loc string) []compute.VirtualMachineScaleSet {
	return []compute.VirtualMachineScaleSet{
		{
			Name: to.StringPtr(name),
			Sku: &compute.Sku{
				Capacity: to.Int64Ptr(cap),
				Name:     to.StringPtr("Standard_D4_v2"),
			},
			VirtualMachineScaleSetProperties: &compute.VirtualMachineScaleSetProperties{},
			Location:                         to.StringPtr(loc),
		},
	}
}

func newTestVMSSVMList(count int) []compute.VirtualMachineScaleSetVM {
	var vmssVMList []compute.VirtualMachineScaleSetVM
	for i := 0; i < count; i++ {
		vmssVM := compute.VirtualMachineScaleSetVM{
			ID:         to.StringPtr(fmt.Sprintf(fakeVirtualMachineScaleSetVMID, i)),
			InstanceID: to.StringPtr(fmt.Sprintf("%d", i)),
			VirtualMachineScaleSetVMProperties: &compute.VirtualMachineScaleSetVMProperties{
				VMID: to.StringPtr(fmt.Sprintf("123E4567-E89B-12D3-A456-426655440000-%d", i)),
			},
		}
		vmssVMList = append(vmssVMList, vmssVM)
	}
	return vmssVMList
}

func TestMaxSize(t *testing.T) {
	provider := newTestProvider(t)
	registered := provider.azureManager.RegisterNodeGroup(
		newTestScaleSet(provider.azureManager, "test-asg"))
	assert.True(t, registered)
	assert.Equal(t, len(provider.NodeGroups()), 1)
	assert.Equal(t, provider.NodeGroups()[0].MaxSize(), 5)
}

func TestMinSize(t *testing.T) {
	provider := newTestProvider(t)
	registered := provider.azureManager.RegisterNodeGroup(
		newTestScaleSet(provider.azureManager, "test-asg"))
	assert.True(t, registered)
	assert.Equal(t, len(provider.NodeGroups()), 1)
	assert.Equal(t, provider.NodeGroups()[0].MinSize(), 1)
}

func TestTargetSize(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	expectedScaleSets := newTestVMSSList(3, "test-asg", "eastus")
	expectedVMSSVMs := newTestVMSSVMList(3)

	provider := newTestProvider(t)
	mockVMSSClient := mockvmssclient.NewMockInterface(ctrl)
	mockVMSSClient.EXPECT().List(gomock.Any(), provider.azureManager.config.ResourceGroup).Return(expectedScaleSets, nil).AnyTimes()
	provider.azureManager.azClient.virtualMachineScaleSetsClient = mockVMSSClient
	mockVMSSVMClient := mockvmssvmclient.NewMockInterface(ctrl)
	mockVMSSVMClient.EXPECT().List(gomock.Any(), provider.azureManager.config.ResourceGroup, "test-asg", gomock.Any()).Return(expectedVMSSVMs, nil).AnyTimes()
	provider.azureManager.azClient.virtualMachineScaleSetVMsClient = mockVMSSVMClient
	err := provider.azureManager.forceRefresh()
	assert.NoError(t, err)

	registered := provider.azureManager.RegisterNodeGroup(
		newTestScaleSet(provider.azureManager, "test-asg"))
	assert.True(t, registered)
	assert.Equal(t, len(provider.NodeGroups()), 1)

	targetSize, err := provider.NodeGroups()[0].TargetSize()
	assert.NoError(t, err)
	assert.Equal(t, 3, targetSize)
}

func TestIncreaseSize(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	expectedScaleSets := newTestVMSSList(3, "test-asg", "eastus")
	expectedVMSSVMs := newTestVMSSVMList(3)

	provider := newTestProvider(t)
	mockVMSSClient := mockvmssclient.NewMockInterface(ctrl)
	mockVMSSClient.EXPECT().List(gomock.Any(), provider.azureManager.config.ResourceGroup).Return(expectedScaleSets, nil).AnyTimes()
	mockVMSSClient.EXPECT().CreateOrUpdateAsync(gomock.Any(), provider.azureManager.config.ResourceGroup, "test-asg", gomock.Any()).Return(nil, nil)
	mockVMSSClient.EXPECT().WaitForAsyncOperationResult(gomock.Any(), gomock.Any()).Return(&http.Response{StatusCode: http.StatusOK}, nil).AnyTimes()
	provider.azureManager.azClient.virtualMachineScaleSetsClient = mockVMSSClient
	mockVMSSVMClient := mockvmssvmclient.NewMockInterface(ctrl)
	mockVMSSVMClient.EXPECT().List(gomock.Any(), provider.azureManager.config.ResourceGroup, "test-asg", gomock.Any()).Return(expectedVMSSVMs, nil).AnyTimes()
	provider.azureManager.azClient.virtualMachineScaleSetVMsClient = mockVMSSVMClient
	err := provider.azureManager.forceRefresh()
	assert.NoError(t, err)

	ss := newTestScaleSet(provider.azureManager, "test-asg")
	ss.lastSizeRefresh = time.Now()
	ss.sizeRefreshPeriod = 1 * time.Minute
	ss.curSize = -1
	err = ss.IncreaseSize(100)
	expectedErr := fmt.Errorf("the scale set test-asg is under initialization, skipping IncreaseSize")
	assert.Equal(t, expectedErr, err)

	registered := provider.azureManager.RegisterNodeGroup(
		newTestScaleSet(provider.azureManager, "test-asg"))
	assert.True(t, registered)
	assert.Equal(t, len(provider.NodeGroups()), 1)

	// current target size is 2.
	targetSize, err := provider.NodeGroups()[0].TargetSize()
	assert.NoError(t, err)
	assert.Equal(t, 3, targetSize)

	// increase 3 nodes.
	err = provider.NodeGroups()[0].IncreaseSize(2)
	assert.NoError(t, err)

	// new target size should be 5.
	targetSize, err = provider.NodeGroups()[0].TargetSize()
	assert.NoError(t, err)
	assert.Equal(t, 5, targetSize)
}

func TestIncreaseSizeOnVMSSUpdating(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	manager := newTestAzureManager(t)
	vmssName := "vmss-updating"
	var vmssCapacity int64 = 3

	expectedScaleSets := []compute.VirtualMachineScaleSet{
		{
			Name: &vmssName,
			Sku: &compute.Sku{
				Capacity: &vmssCapacity,
			},
			VirtualMachineScaleSetProperties: &compute.VirtualMachineScaleSetProperties{
				ProvisioningState: to.StringPtr(string(compute.ProvisioningStateUpdating)),
			},
		},
	}
	expectedVMSSVMs := newTestVMSSVMList(3)

	mockVMSSClient := mockvmssclient.NewMockInterface(ctrl)
	mockVMSSClient.EXPECT().List(gomock.Any(), manager.config.ResourceGroup).Return(expectedScaleSets, nil)
	mockVMSSClient.EXPECT().CreateOrUpdateAsync(gomock.Any(), manager.config.ResourceGroup, vmssName, gomock.Any()).Return(nil, nil)
	mockVMSSClient.EXPECT().WaitForAsyncOperationResult(gomock.Any(), gomock.Any()).Return(&http.Response{StatusCode: http.StatusOK}, nil).AnyTimes()
	manager.azClient.virtualMachineScaleSetsClient = mockVMSSClient
	mockVMSSVMClient := mockvmssvmclient.NewMockInterface(ctrl)
	mockVMSSVMClient.EXPECT().List(gomock.Any(), manager.config.ResourceGroup, "vmss-updating", gomock.Any()).Return(expectedVMSSVMs, nil).AnyTimes()
	manager.azClient.virtualMachineScaleSetVMsClient = mockVMSSVMClient
	manager.explicitlyConfigured["vmss-updating"] = true
	registered := manager.RegisterNodeGroup(newTestScaleSet(manager, vmssName))
	assert.True(t, registered)
	manager.Refresh()

	provider, err := BuildAzureCloudProvider(manager, nil)
	assert.NoError(t, err)

	// Scaling should continue even VMSS is under updating.
	scaleSet, ok := provider.NodeGroups()[0].(*ScaleSet)
	assert.True(t, ok)
	err = scaleSet.IncreaseSize(1)
	assert.NoError(t, err)
}

func TestBelongs(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	expectedScaleSets := newTestVMSSList(3, "test-asg", "eastus")
	expectedVMSSVMs := newTestVMSSVMList(3)

	provider := newTestProvider(t)
	mockVMSSClient := mockvmssclient.NewMockInterface(ctrl)
	mockVMSSClient.EXPECT().List(gomock.Any(), provider.azureManager.config.ResourceGroup).Return(expectedScaleSets, nil)
	provider.azureManager.azClient.virtualMachineScaleSetsClient = mockVMSSClient
	mockVMSSVMClient := mockvmssvmclient.NewMockInterface(ctrl)
	mockVMSSVMClient.EXPECT().List(gomock.Any(), provider.azureManager.config.ResourceGroup, "test-asg", gomock.Any()).Return(expectedVMSSVMs, nil).AnyTimes()
	provider.azureManager.azClient.virtualMachineScaleSetVMsClient = mockVMSSVMClient

	registered := provider.azureManager.RegisterNodeGroup(
		newTestScaleSet(provider.azureManager, "test-asg"))
	assert.True(t, registered)

	scaleSet, ok := provider.NodeGroups()[0].(*ScaleSet)
	assert.True(t, ok)
	provider.azureManager.explicitlyConfigured["test-asg"] = true
	provider.azureManager.Refresh()

	invalidNode := &apiv1.Node{
		Spec: apiv1.NodeSpec{
			ProviderID: "azure:///subscriptions/test-subscrition-id/resourcegroups/invalid-asg/providers/microsoft.compute/virtualmachinescalesets/agents/virtualmachines/0",
		},
	}
	_, err := scaleSet.Belongs(invalidNode)
	assert.Error(t, err)

	validNode := &apiv1.Node{
		Spec: apiv1.NodeSpec{
			ProviderID: "azure://" + fmt.Sprintf(fakeVirtualMachineScaleSetVMID, 0),
		},
	}
	belongs, err := scaleSet.Belongs(validNode)
	assert.Equal(t, true, belongs)
	assert.NoError(t, err)
}

func TestDeleteNodes(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	manager := newTestAzureManager(t)
	vmssName := "test-asg"
	var vmssCapacity int64 = 3

	expectedScaleSets := []compute.VirtualMachineScaleSet{
		{
			Name: &vmssName,
			Sku: &compute.Sku{
				Capacity: &vmssCapacity,
			},
		},
	}
	expectedVMSSVMs := newTestVMSSVMList(3)

	mockVMSSClient := mockvmssclient.NewMockInterface(ctrl)
	mockVMSSClient.EXPECT().List(gomock.Any(), manager.config.ResourceGroup).Return(expectedScaleSets, nil).Times(2)
	mockVMSSClient.EXPECT().DeleteInstancesAsync(gomock.Any(), manager.config.ResourceGroup, gomock.Any(), gomock.Any()).Return(nil, nil)
	mockVMSSClient.EXPECT().WaitForAsyncOperationResult(gomock.Any(), gomock.Any()).Return(&http.Response{StatusCode: http.StatusOK}, nil).AnyTimes()
	manager.azClient.virtualMachineScaleSetsClient = mockVMSSClient
	mockVMSSVMClient := mockvmssvmclient.NewMockInterface(ctrl)
	mockVMSSVMClient.EXPECT().List(gomock.Any(), manager.config.ResourceGroup, "test-asg", gomock.Any()).Return(expectedVMSSVMs, nil).AnyTimes()
	manager.azClient.virtualMachineScaleSetVMsClient = mockVMSSVMClient
	err := manager.forceRefresh()
	assert.NoError(t, err)

	resourceLimiter := cloudprovider.NewResourceLimiter(
		map[string]int64{cloudprovider.ResourceNameCores: 1, cloudprovider.ResourceNameMemory: 10000000},
		map[string]int64{cloudprovider.ResourceNameCores: 10, cloudprovider.ResourceNameMemory: 100000000})
	provider, err := BuildAzureCloudProvider(manager, resourceLimiter)
	assert.NoError(t, err)

	registered := manager.RegisterNodeGroup(
		newTestScaleSet(manager, "test-asg"))
	manager.explicitlyConfigured["test-asg"] = true
	assert.True(t, registered)
	err = manager.forceRefresh()
	assert.NoError(t, err)

	scaleSet, ok := provider.NodeGroups()[0].(*ScaleSet)
	assert.True(t, ok)

	targetSize, err := scaleSet.TargetSize()
	assert.NoError(t, err)
	assert.Equal(t, 3, targetSize)

	// Perform the delete operation
	nodesToDelete := []*apiv1.Node{
		{
			Spec: apiv1.NodeSpec{
				ProviderID: "azure://" + fmt.Sprintf(fakeVirtualMachineScaleSetVMID, 0),
			},
		},
		{
			Spec: apiv1.NodeSpec{
				ProviderID: "azure://" + fmt.Sprintf(fakeVirtualMachineScaleSetVMID, 2),
			},
		},
	}
	err = scaleSet.DeleteNodes(nodesToDelete)
	assert.NoError(t, err)
	vmssCapacity = 1
	expectedScaleSets = []compute.VirtualMachineScaleSet{
		{
			Name: &vmssName,
			Sku: &compute.Sku{
				Capacity: &vmssCapacity,
			},
		},
	}
	mockVMSSClient.EXPECT().List(gomock.Any(), manager.config.ResourceGroup).Return(expectedScaleSets, nil).AnyTimes()
	expectedVMSSVMs[0].ProvisioningState = to.StringPtr(string(compute.ProvisioningStateDeleting))
	expectedVMSSVMs[2].ProvisioningState = to.StringPtr(string(compute.ProvisioningStateDeleting))
	mockVMSSVMClient.EXPECT().List(gomock.Any(), manager.config.ResourceGroup, "test-asg", gomock.Any()).Return(expectedVMSSVMs, nil).AnyTimes()
	err = manager.forceRefresh()
	assert.NoError(t, err)

	// Ensure the the cached size has been proactively decremented by 2
	targetSize, err = scaleSet.TargetSize()
	assert.NoError(t, err)
	assert.Equal(t, 1, targetSize)

	// Ensure that the status for the instances is Deleting
	instance0, found := scaleSet.getInstanceByProviderID("azure://" + fmt.Sprintf(fakeVirtualMachineScaleSetVMID, 0))
	assert.True(t, found, true)
	assert.Equal(t, instance0.Status.State, cloudprovider.InstanceDeleting)

	instance2, found := scaleSet.getInstanceByProviderID("azure://" + fmt.Sprintf(fakeVirtualMachineScaleSetVMID, 2))
	assert.True(t, found, true)
	assert.Equal(t, instance2.Status.State, cloudprovider.InstanceDeleting)
}

func TestDeleteNodeUnregistered(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	manager := newTestAzureManager(t)
	vmssName := "test-asg"
	var vmssCapacity int64 = 2

	expectedScaleSets := []compute.VirtualMachineScaleSet{
		{
			Name: &vmssName,
			Sku: &compute.Sku{
				Capacity: &vmssCapacity,
			},
		},
	}
	expectedVMSSVMs := newTestVMSSVMList(2)

	mockVMSSClient := mockvmssclient.NewMockInterface(ctrl)
	mockVMSSClient.EXPECT().List(gomock.Any(), manager.config.ResourceGroup).Return(expectedScaleSets, nil).Times(2)
	mockVMSSClient.EXPECT().DeleteInstancesAsync(gomock.Any(), manager.config.ResourceGroup, gomock.Any(), gomock.Any()).Return(nil, nil)
	mockVMSSClient.EXPECT().WaitForAsyncOperationResult(gomock.Any(), gomock.Any()).Return(&http.Response{StatusCode: http.StatusOK}, nil).AnyTimes()
	manager.azClient.virtualMachineScaleSetsClient = mockVMSSClient
	mockVMSSVMClient := mockvmssvmclient.NewMockInterface(ctrl)
	mockVMSSVMClient.EXPECT().List(gomock.Any(), manager.config.ResourceGroup, "test-asg", gomock.Any()).Return(expectedVMSSVMs, nil).AnyTimes()
	manager.azClient.virtualMachineScaleSetVMsClient = mockVMSSVMClient
	err := manager.forceRefresh()
	assert.NoError(t, err)

	resourceLimiter := cloudprovider.NewResourceLimiter(
		map[string]int64{cloudprovider.ResourceNameCores: 1, cloudprovider.ResourceNameMemory: 10000000},
		map[string]int64{cloudprovider.ResourceNameCores: 10, cloudprovider.ResourceNameMemory: 100000000})
	provider, err := BuildAzureCloudProvider(manager, resourceLimiter)
	assert.NoError(t, err)

	registered := manager.RegisterNodeGroup(
		newTestScaleSet(manager, "test-asg"))
	manager.explicitlyConfigured["test-asg"] = true
	assert.True(t, registered)
	err = manager.forceRefresh()
	assert.NoError(t, err)

	scaleSet, ok := provider.NodeGroups()[0].(*ScaleSet)
	assert.True(t, ok)

	targetSize, err := scaleSet.TargetSize()
	assert.NoError(t, err)
	assert.Equal(t, 2, targetSize)

	// annotate node with unregistered annotation
	annotations := make(map[string]string)
	annotations[cloudprovider.FakeNodeReasonAnnotation] = cloudprovider.FakeNodeUnregistered
	nodesToDelete := []*apiv1.Node{
		{
			ObjectMeta: metav1.ObjectMeta{
				Annotations: annotations,
			},
			Spec: apiv1.NodeSpec{
				ProviderID: "azure://" + fmt.Sprintf(fakeVirtualMachineScaleSetVMID, 0),
			},
		},
	}
	err = scaleSet.DeleteNodes(nodesToDelete)
	assert.NoError(t, err)

	// Ensure the the cached size has NOT been proactively decremented
	targetSize, err = scaleSet.TargetSize()
	assert.NoError(t, err)
	assert.Equal(t, 2, targetSize)

	// Ensure that the status for the instances is Deleting
	instance0, found := scaleSet.getInstanceByProviderID("azure://" + fmt.Sprintf(fakeVirtualMachineScaleSetVMID, 0))
	assert.True(t, found, true)
	assert.Equal(t, instance0.Status.State, cloudprovider.InstanceDeleting)
}

func TestDeleteNoConflictRequest(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	vmssName := "test-asg"
	var vmssCapacity int64 = 3

	manager := newTestAzureManager(t)

	expectedVMSSVMs := []compute.VirtualMachineScaleSetVM{
		{
			ID:         to.StringPtr(fakeVirtualMachineScaleSetVMID),
			InstanceID: to.StringPtr("0"),
			VirtualMachineScaleSetVMProperties: &compute.VirtualMachineScaleSetVMProperties{
				VMID:              to.StringPtr("123E4567-E89B-12D3-A456-426655440000"),
				ProvisioningState: to.StringPtr("Deleting"),
			},
		},
	}
	expectedScaleSets := []compute.VirtualMachineScaleSet{
		{
			Name: &vmssName,
			Sku: &compute.Sku{
				Capacity: &vmssCapacity,
			},
		},
	}

	mockVMSSClient := mockvmssclient.NewMockInterface(ctrl)
	mockVMSSClient.EXPECT().List(gomock.Any(), manager.config.ResourceGroup).Return(expectedScaleSets, nil).AnyTimes()
	manager.azClient.virtualMachineScaleSetsClient = mockVMSSClient
	mockVMSSVMClient := mockvmssvmclient.NewMockInterface(ctrl)
	mockVMSSVMClient.EXPECT().List(gomock.Any(), manager.config.ResourceGroup, "test-asg", gomock.Any()).Return(expectedVMSSVMs, nil).AnyTimes()
	manager.azClient.virtualMachineScaleSetVMsClient = mockVMSSVMClient

	resourceLimiter := cloudprovider.NewResourceLimiter(
		map[string]int64{cloudprovider.ResourceNameCores: 1, cloudprovider.ResourceNameMemory: 10000000},
		map[string]int64{cloudprovider.ResourceNameCores: 10, cloudprovider.ResourceNameMemory: 100000000})
	provider, err := BuildAzureCloudProvider(manager, resourceLimiter)
	assert.NoError(t, err)

	registered := manager.RegisterNodeGroup(newTestScaleSet(manager, "test-asg"))
	manager.explicitlyConfigured["test-asg"] = true
	assert.True(t, registered)
	manager.Refresh()

	node := &apiv1.Node{
		Spec: apiv1.NodeSpec{
			ProviderID: "azure://" + fmt.Sprintf(fakeVirtualMachineScaleSetVMID, 0),
		},
	}

	scaleSet, ok := provider.NodeGroups()[0].(*ScaleSet)
	assert.True(t, ok)

	err = scaleSet.DeleteNodes([]*apiv1.Node{node})
}

func TestId(t *testing.T) {
	provider := newTestProvider(t)
	registered := provider.azureManager.RegisterNodeGroup(
		newTestScaleSet(provider.azureManager, "test-asg"))
	assert.True(t, registered)
	assert.Equal(t, len(provider.NodeGroups()), 1)
	assert.Equal(t, provider.NodeGroups()[0].Id(), "test-asg")
}

func TestDebug(t *testing.T) {
	asg := ScaleSet{
		manager: newTestAzureManager(t),
		minSize: 5,
		maxSize: 55,
	}
	asg.Name = "test-scale-set"
	assert.Equal(t, asg.Debug(), "test-scale-set (5:55)")
}

func TestScaleSetNodes(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	expectedVMSSVMs := newTestVMSSVMList(3)
	expectedScaleSets := newTestVMSSList(3, "test-asg", "eastus")

	provider := newTestProvider(t)
	mockVMSSClient := mockvmssclient.NewMockInterface(ctrl)
	mockVMSSClient.EXPECT().List(gomock.Any(), provider.azureManager.config.ResourceGroup).Return(expectedScaleSets, nil).AnyTimes()
	provider.azureManager.azClient.virtualMachineScaleSetsClient = mockVMSSClient
	mockVMSSVMClient := mockvmssvmclient.NewMockInterface(ctrl)
	mockVMSSVMClient.EXPECT().List(gomock.Any(), provider.azureManager.config.ResourceGroup, "test-asg", gomock.Any()).Return(expectedVMSSVMs, nil).AnyTimes()
	provider.azureManager.azClient.virtualMachineScaleSetVMsClient = mockVMSSVMClient

	registered := provider.azureManager.RegisterNodeGroup(
		newTestScaleSet(provider.azureManager, "test-asg"))
	provider.azureManager.explicitlyConfigured["test-asg"] = true
	provider.azureManager.Refresh()
	assert.True(t, registered)
	assert.Equal(t, len(provider.NodeGroups()), 1)

	node := &apiv1.Node{
		Spec: apiv1.NodeSpec{
			ProviderID: "azure://" + fmt.Sprintf(fakeVirtualMachineScaleSetVMID, 0),
		},
	}
	group, err := provider.NodeGroupForNode(node)
	assert.NoError(t, err)
	assert.NotNil(t, group, "Group should not be nil")
	assert.Equal(t, group.Id(), "test-asg")
	assert.Equal(t, group.MinSize(), 1)
	assert.Equal(t, group.MaxSize(), 5)

	ss, ok := group.(*ScaleSet)
	assert.True(t, ok)
	assert.NotNil(t, ss)
	instances, err := group.Nodes()
	assert.NoError(t, err)
	assert.Equal(t, len(instances), 3)
	assert.Equal(t, instances[0], cloudprovider.Instance{Id: "azure://" + fmt.Sprintf(fakeVirtualMachineScaleSetVMID, 0)})
	assert.Equal(t, instances[1], cloudprovider.Instance{Id: "azure://" + fmt.Sprintf(fakeVirtualMachineScaleSetVMID, 1)})
	assert.Equal(t, instances[2], cloudprovider.Instance{Id: "azure://" + fmt.Sprintf(fakeVirtualMachineScaleSetVMID, 2)})
}

func TestTemplateNodeInfo(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	expectedScaleSets := newTestVMSSList(3, "test-asg", "eastus")

	provider := newTestProvider(t)
	mockVMSSClient := mockvmssclient.NewMockInterface(ctrl)
	mockVMSSClient.EXPECT().List(gomock.Any(), provider.azureManager.config.ResourceGroup).Return(expectedScaleSets, nil).AnyTimes()
	provider.azureManager.azClient.virtualMachineScaleSetsClient = mockVMSSClient
	err := provider.azureManager.forceRefresh()
	assert.NoError(t, err)

	registered := provider.azureManager.RegisterNodeGroup(
		newTestScaleSet(provider.azureManager, "test-asg"))
	assert.True(t, registered)
	assert.Equal(t, len(provider.NodeGroups()), 1)

	asg := ScaleSet{
		manager: provider.azureManager,
		minSize: 1,
		maxSize: 5,
	}
	asg.Name = "test-asg"

	nodeInfo, err := asg.TemplateNodeInfo()
	assert.NoError(t, err)
	assert.NotNil(t, nodeInfo)
	assert.NotEmpty(t, nodeInfo.Pods)
}
