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

package pod

import (
	"k8s.io/kubernetes/pkg/kubelet/types"

	apiv1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	// DaemonSetPodAnnotationKey - annotation use to informs the cluster-autoscaler controller when a pod needs to be considered as a Daemonset's Pod.
	DaemonSetPodAnnotationKey = "cluster-autoscaler.kubernetes.io/daemonset-pod"
)

// IsDaemonSetPod returns true if the Pod should be considered as Pod managed by a DaemonSet
func IsDaemonSetPod(pod *apiv1.Pod) bool {
	controllerRef := metav1.GetControllerOf(pod)
	if controllerRef != nil && controllerRef.Kind == "DaemonSet" {
		return true
	}

	if val, ok := pod.Annotations[DaemonSetPodAnnotationKey]; ok && val == "true" {
		return true
	}

	return false
}

// IsMirrorPod checks whether the pod is a mirror pod.
func IsMirrorPod(pod *apiv1.Pod) bool {
	if pod.ObjectMeta.Annotations == nil {
		return false
	}
	_, found := pod.ObjectMeta.Annotations[types.ConfigMirrorAnnotationKey]
	return found
}

// IsStaticPod returns true if the pod is a static pod.
func IsStaticPod(pod *apiv1.Pod) bool {
	if pod.Annotations != nil {
		if source, ok := pod.Annotations[types.ConfigSourceAnnotationKey]; ok == true {
			return source != types.ApiserverSource
		}
	}
	return false
}
